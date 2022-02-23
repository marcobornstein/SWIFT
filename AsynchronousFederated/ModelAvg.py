import numpy as np
import os
from mpi4py import MPI
import torch
import util
from comm_helpers import flatten_tensors, unflatten_tensors


def model_avg(worker_size, model, test_data, args):

    consensus_accuracy = list()
    tensor_list = list()
    for param in model.parameters():
        tensor_list.append(param)
    send_buffer = flatten_tensors(tensor_list).cpu()
    model.eval()

    for epoch in range(args.epoch):

        # Clear the buffers
        avg_model = torch.zeros_like(send_buffer)

        # Get weighting (build a function) -- For now, make it uniform
        weighting = (1/worker_size) * np.ones(worker_size)

        for rank in range(worker_size):
            worker_model = np.empty_like(avg_model)
            MPI.COMM_WORLD.Recv(worker_model, source=rank, tag=rank+10*worker_size)
            avg_model.add_(torch.from_numpy(worker_model), alpha=weighting[rank])

        reset_model(avg_model, tensor_list)

        # model.train()
        # Start training each epoch
        accuracy = AverageMeter()
        for batch_idx, (data, target) in enumerate(test_data):
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = model(data)
            acc1 = util.comp_accuracy(output, target)
            accuracy.update(acc1[0].item(), data.size(0))

        test_acc = accuracy.avg

        consensus_accuracy.append(test_acc)
        print('Consensus Accuracy for Epoch %d is %.3f' % (epoch, test_acc))

    subfolder = args.outputFolder + '/run-' + args.name + '-' + str(args.epoch) + 'epochs'

    isExist = os.path.exists(subfolder)
    if not isExist:
        os.makedirs(subfolder)

    np.savetxt(subfolder + '/consensus-average-' + args.comm_style + '-.log', consensus_accuracy, delimiter=',')

    with open(subfolder + '/ExpDescription', 'w') as f:
        f.write(str(args) + '\n')
        f.write(args.description + '\n')


def reset_model(avg_model, tensor_list):
    # Reset local models to be the averaged model
    for f, t in zip(unflatten_tensors(avg_model.cuda(), tensor_list), tensor_list):
        with torch.no_grad():
            t.set_(f)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res