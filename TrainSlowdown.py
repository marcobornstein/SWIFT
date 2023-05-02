import numpy as np
import time
import argparse
from GDM import Resnet
from GDM.GraphConstruct import GraphConstruct
from Communicators.AsyncCommunicator import AsyncDecentralized
from Communicators.DSGD import decenCommunicator
from mpi4py import MPI
from GDM.DataPartition import partition_dataset
from Communicators.CommHelpers import flatten_tensors
from Utils.Misc import AverageMeter, Recorder, test_accuracy, test_loss, compute_accuracy

import torch
import torch.utils.data.distributed
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def run(rank, size):

    # set random seed
    torch.manual_seed(args.randomSeed + rank)
    np.random.seed(args.randomSeed)

    # select neural network model
    num_class = 10
    model = Resnet.ResNet(args.resSize, num_class)

    # split up GPUs
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus

    # initialize the GPU being used
    torch.cuda.set_device(gpu_id)
    model = model.cuda(gpu_id)

    # model loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=1e-4,
                          nesterov=args.nesterov)

    # guarantee all local models start from the same point
    init_model = sync_allreduce(model, size, MPI.COMM_WORLD)

    # load data
    val_split = 0
    train_loader, test_loader = partition_dataset(rank, size, MPI.COMM_WORLD, val_split, args)

    # ensure swift uses its own weighting
    if args.comm_style == 'swift':
        args.weight_type = 'swift'

    # load base network topology
    p = 3/size
    GP = GraphConstruct(rank, size, MPI.COMM_WORLD, args.graph, args.weight_type, p=p, num_c=args.num_clusters)

    if args.comm_style == 'swift':
        communicator = AsyncDecentralized(rank, size, MPI.COMM_WORLD, GP,
                                          args.sgd_steps, args.max_sgd, args.wb, args.memory_efficient, init_model)
    elif args.comm_style == 'ld-sgd':
        communicator = decenCommunicator(rank, size, MPI.COMM_WORLD, GP, args.i1, args.i2)
    elif args.comm_style == 'pd-sgd':
        communicator = decenCommunicator(rank, size, MPI.COMM_WORLD, GP, args.i1, 1)
    elif args.comm_style == 'd-sgd':
        communicator = decenCommunicator(rank, size, MPI.COMM_WORLD, GP, 0, 1)
    else:
        # Anything else just default to our algorithm
        communicator = AsyncDecentralized(rank, size, MPI.COMM_WORLD, GP,
                                          args.sgd_steps, args.max_sgd, args.wb, args.memory_efficient, init_model)

    # init recorder
    comp_time = 0
    comm_time = 0
    recorder = Recorder(args, rank)
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.noniid:
        d_epoch = 200
    else:
        d_epoch = 50

    MPI.COMM_WORLD.Barrier()
    # start training
    for epoch in range(args.epoch):
        init_time = time.time()
        record_time = 0
        model.train()

        # Start training each epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            start_time = time.time()
            # data loading
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # forward pass
            output = model(data)
            loss = criterion(output, target)

            # record training loss and accuracy
            record_start = time.time()
            acc1 = compute_accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0].item(), data.size(0))
            record_end = time.time() - record_start
            record_time += record_end

            # backward pass
            loss.backward()

            # communication happens here
            comm_start = time.time()
            d_comm_time = communicator.communicate(model)
            comm_t = time.time() - comm_start

            # gradient step
            optimizer.step()
            optimizer.zero_grad()
            end_time = time.time()

            # compute computational time
            comp_time += (end_time - start_time - comm_t)

            # compute communication time
            comm_time += d_comm_time

            # slowdown
            if rank == 0 and args.slowdown > 1:
                sleep_time = (args.slowdown - 1) * (end_time - start_time - comm_t)
                time.sleep(sleep_time)

        # update learning rate here
        if not args.customLR:
            update_learning_rate(optimizer, epoch, drop=0.5, epochs_drop=10.0, decay_epoch=d_epoch,
                                    itr_per_epoch=len(train_loader))
        else:
            if epoch == 81 or epoch == 122:
                args.lr *= 0.1
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.lr

        # evaluate test accuracy at the end of each epoch
        t = time.time()
        t_loss = test_loss(model, test_loader, criterion)
        test_time = time.time() - t

        # total time spent in algorithm
        comp_time -= record_time
        epoch_time = comp_time + comm_time

        print("rank: %d, epoch: %.3f, loss: %.3f, train_acc: %.3f, test_loss: %.3f, comp time: %.3f, "
              "epoch time: %.3f" % (rank, epoch, losses.avg, top1.avg, t_loss, comp_time, epoch_time))

        recorder.add_new(comp_time, comm_time, epoch_time, (time.time() - init_time)-test_time,
                         top1.avg, losses.avg, t_loss)

        # reset recorders
        comp_time, comm_time = 0, 0
        losses.reset()
        top1.reset()

    # Save data to output folder
    recorder.save_to_file()

    # Broadcast/wait until all other neighbors are finished in async algorithm
    if args.comm_style == 'swift' and args.memory_efficient:
        communicator.wait(model)
        print('Finished from Rank %d' % rank)

    MPI.COMM_WORLD.Barrier()

    sync_allreduce(model, size, MPI.COMM_WORLD)
    test_acc = test_accuracy(model, test_loader)
    print("rank %d: Test Accuracy %.3f" % (rank, test_acc))


def update_learning_rate(optimizer, epoch, drop, epochs_drop, decay_epoch, itr=None, itr_per_epoch=None):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially starting at decay_epoch
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    base_lr = 0.1
    lr = args.lr

    if args.warmup and epoch < 5:  # warmup to scaled lr
        if lr > base_lr:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (lr - base_lr) * (count / (5 * itr_per_epoch))
            lr = base_lr + incr
    elif epoch >= decay_epoch:
        lr *= np.power(drop, np.floor((1 + epoch - decay_epoch) / epochs_drop))

    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def sync_allreduce(model, size, comm):
    senddata = {}
    recvdata = {}
    for param in model.parameters():
        tmp = param.data.cpu()
        senddata[param] = tmp.numpy()
        recvdata[param] = np.empty(senddata[param].shape, dtype=senddata[param].dtype)
    torch.cuda.synchronize()
    comm.Barrier()

    for param in model.parameters():
        comm.Allreduce(senddata[param], recvdata[param], op=MPI.SUM)
    torch.cuda.synchronize()
    comm.Barrier()

    tensor_list = list()
    for param in model.parameters():
        tensor_list.append(param)
        param.data = torch.Tensor(recvdata[param]).cuda()
        param.data = param.data / float(size)

    # flatten tensors
    initial_model = flatten_tensors(tensor_list).cpu().detach().numpy()

    return initial_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--name', '-n', default="default", type=str, help='experiment name')
    parser.add_argument('--description', type=str, help='experiment description')

    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--comm_style', default='swift', type=str, help='baseline communicator')
    parser.add_argument('--resSize', default=50, type=int, help='res net size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate to start from \
                        (if not customLR then lr always 0.1)')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--epoch', '-e', default=10, type=int, help='total epoch')
    parser.add_argument('--bs', default=64, type=int, help='batch size on each worker')
    parser.add_argument('--noniid', default=1, type=int, help='use non iid data or not')
    parser.add_argument('--degree_noniid', default=0.7, type=float, help='how distributed are labels (0 is random)')
    parser.add_argument('--weight_type', default='uniform', type=str, help='how do workers average with each other')
    parser.add_argument('--unordered_epochs', default=1, type=int, help='calculate consensus after the first n models')
    parser.add_argument('--slowdown', default=1, type=int, help='perform slowdown testing (default is no)')

    # Specific async arguments
    parser.add_argument('--wb', default=0, type=int, help='proportionally increase neighbor weights or self replace')
    parser.add_argument('--memory_efficient', default=0, type=int, help='DO store all neighbor local models')
    parser.add_argument('--max_sgd', default=10, type=int, help='max sgd steps per worker')
    parser.add_argument('--personalize', default=0, type=int, help='use personalization or not')

    parser.add_argument('--i1', default=0, type=int, help='i1 comm set, number of local updates no averaging')
    parser.add_argument('--i2', default=1, type=int, help='i2 comm set, number of d-sgd updates')
    parser.add_argument('--sgd_steps', default=1, type=int, help='baseline sgd steps per worker')
    parser.add_argument('--num_clusters', default=1, type=int, help='number of clusters in graph')
    parser.add_argument('--graph', type=str, help='graph topology')

    parser.add_argument('--warmup', action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov momentum or not')
    parser.add_argument('--dataset', default='cifar10', type=str, help='the dataset')
    parser.add_argument('--datasetRoot', type=str, help='the path of dataset')
    parser.add_argument('--downloadCifar', default=0, type=int, help='change to 1 if needing to download Cifar')
    parser.add_argument('--p', '-p', action='store_true', help='partition the dataset or not')
    parser.add_argument('--savePath', type=str, help='save path')
    parser.add_argument('--outputFolder', type=str, help='save folder')
    parser.add_argument('--randomSeed', default=9001, type=int, help='random seed')
    parser.add_argument('--customLR', default=0, type=int, help='custom learning rate strategy, 1 if using multi-step')

    args = parser.parse_args()
    print(args.datasetRoot)

    if not args.description:
        print('No experiment description, exit!')
        exit()

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    run(rank, size)
