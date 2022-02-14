import numpy as np
import time
import argparse
import resnet
import util
from GraphConstruct import GraphConstruct
from AsyncCommunicator import AsyncDecentralized
from mpi4py import MPI

import torch
import torch.utils.data.distributed
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def run(rank, size):

    # set random seed
    torch.manual_seed(args.randomSeed+rank)
    np.random.seed(args.randomSeed)

    # load data
    train_loader, test_loader = util.partition_dataset(rank, size, args)    
    # num_batches = ceil(len(train_loader.dataset) / float(args.bs))

    # load base network topology
    GP = GraphConstruct('clique-ring', rank, size, num_c=2)
    sgd_steps = 3
    communicator = AsyncDecentralized(rank, size, GP, sgd_steps, args.max_sgd)

    # select neural network model
    num_class = 10
    model = resnet.ResNet(args.resSize, num_class)

    # split up GPUs                                                                                                                                              
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    
    # initialize the GPU being used                                                                                                                              
    torch.cuda.set_device(gpu_id)

    model = model.cuda(gpu_id)
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr,
                          momentum=args.momentum, 
                          weight_decay=5e-4,
                          nesterov=args.nesterov)
    
    # guarantee all local models start from the same point
    sync_allreduce(model, size)

    # init recorder
    comp_time = 0
    comm_time = 0
    recorder = util.Recorder(args, rank)
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    init_time = time.time()

    MPI.COMM_WORLD.Barrier()
    # start training
    for epoch in range(args.epoch):
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
            acc1 = util.comp_accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0].item(), data.size(0))
            record_end = time.time()

            # backward pass
            loss.backward()
            update_learning_rate(optimizer, epoch, drop=0.75, epochs_drop=10.0, decay_epoch=20,
                                 itr=batch_idx, itr_per_epoch=len(train_loader))

            # gradient step
            optimizer.step()
            optimizer.zero_grad()
            end_time = time.time()

            # compute computational time
            d_comp_time = (end_time - start_time - (record_end - record_start))
            comp_time += d_comp_time

            # communication happens here
            d_comm_time = communicator.communicate(model)
            comm_time += d_comm_time

        # evaluate test accuracy at the end of each epoch
        test_acc = util.test(model, test_loader)[0].item()

        comm_time2 = 0
        if args.personalize:
            comm_time2 += communicator.personalize(test_acc)

        total_comm_time = comm_time + comm_time2

        # total time spent in algorithm
        epoch_time = comp_time + total_comm_time

        print("rank: %d, epoch: %.3f, loss: %.3f, train_acc: %.3f, test_acc: %.3f epoch time: %.3f"
              % (rank, epoch, losses.avg, top1.avg, test_acc, epoch_time))

        # if rank == 0:
        #    print("comp_time: %.3f, comm_time: %.3f, comp_time_budget: %.3f, comm_time_budget: %.3f"
        #          % (comp_time, comm_time, comp_time/epoch_time, comm_time/epoch_time))

        recorder.add_new(comp_time, total_comm_time, epoch_time, time.time()-init_time, top1.avg, losses.avg, test_acc)

        # reset recorders
        comp_time, comm_time = 0, 0
        losses.reset()
        top1.reset()

    recorder.save_to_file()


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
        lr *= np.power(drop, np.floor((1 + epoch) / epochs_drop))

    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def sync_allreduce(model, size):
    senddata = {}
    recvdata = {}
    for param in model.parameters():
        tmp = param.data.cpu()
        senddata[param] = tmp.numpy()
        recvdata[param] = np.empty(senddata[param].shape, dtype=senddata[param].dtype)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()

    comm_start = time.time()
    for param in model.parameters():
        MPI.COMM_WORLD.Allreduce(senddata[param], recvdata[param], op=MPI.SUM)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()

    comm_end = time.time()
    comm_t = (comm_end - comm_start)

    for param in model.parameters():
        param.data = torch.Tensor(recvdata[param]).cuda()
        param.data = param.data / float(size)
    return comm_t


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--name', '-n', default="default", type=str, help='experiment name')
    parser.add_argument('--description', type=str, help='experiment description')
    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--resSize', default=50, type=int, help='res net size')
    parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--epoch', '-e', default=1, type=int, help='total epoch')
    parser.add_argument('--bs', default=4, type=int, help='batch size on each worker')
    parser.add_argument('--max_sgd', default=10, type=int, help='max sgd steps per worker')
    parser.add_argument('--personalize', default=1, type=int, help='use personalization or not')
    parser.add_argument('--warmup', action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov momentum or not')
    parser.add_argument('--dataset', default='cifar10', type=str, help='the dataset')
    parser.add_argument('--datasetRoot', type=str, help='the path of dataset')
    parser.add_argument('--downloadCifar', default=0, type=int, help='change to 1 if needing to download Cifar')
    parser.add_argument('--p', '-p', action='store_true', help='partition the dataset or not')
    parser.add_argument('--savePath', type=str, help='save path')
    parser.add_argument('--outputFolder', type=str, help = 'save folder')
    parser.add_argument('--randomSeed', type=int, help='random seed')

    args = parser.parse_args()

    if not args.description:
        print('No experiment description, exit!')
        exit()

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    run(rank, size)
