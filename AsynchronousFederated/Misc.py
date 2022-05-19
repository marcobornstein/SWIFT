import torch
import numpy as np
import os

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


class Recorder(object):
    def __init__(self, args, rank):
        # self.record_valacc = list()
        self.record_timing = list()
        self.record_total_timing = list()
        self.record_comp_timing = list()
        self.record_comm_timing = list()
        self.record_losses = list()
        self.record_trainacc = list()
        self.record_testloss = list()
        self.total_record_timing = list()
        self.args = args
        self.rank = rank
        self.saveFolderName = args.outputFolder + '/' + self.args.name + '-' + str(self.args.graph) + '-' \
                              + str(self.args.sgd_steps) + 'sgd-' + str(self.args.epoch) + 'epochs'
        if rank == 0 and not os.path.isdir(self.saveFolderName):
            os.mkdir(self.saveFolderName)

    def add_new(self, comp_time, comm_time, epoch_time, total_time, top1, losses, test_loss):
        self.record_timing.append(epoch_time)
        self.record_total_timing.append(total_time)
        self.record_comp_timing.append(comp_time)
        self.record_comm_timing.append(comm_time)
        self.record_trainacc.append(top1)
        self.record_losses.append(losses)
        # self.record_valacc.append(val_acc)
        self.record_testloss.append(test_loss)

    def save_to_file(self):
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-epoch-time.log', self.record_timing, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-total-time.log', self.record_total_timing,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comptime.log', self.record_comp_timing,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-commtime.log', self.record_comm_timing,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-losses.log', self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-tacc.log', self.record_trainacc, delimiter=',')
        # np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-vacc.log', self.record_valacc, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-testloss.log', self.record_testloss, delimiter=',')
        with open(self.saveFolderName + '/ExpDescription', 'w') as f:
            f.write(str(self.args) + '\n')
            f.write(self.args.description + '\n')


def compute_accuracy(output, target, topk=(1,)):
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


def test_accuracy(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        acc1 = compute_accuracy(outputs, targets)
        top1.update(acc1[0].item(), inputs.size(0))
    return top1.avg


def test_loss(model, test_loader, criterion):
    model.eval()
    top1 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        top1.update(loss.item(), inputs.size(0))
    return top1.avg

