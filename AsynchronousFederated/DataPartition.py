import numpy as np
import os
from mpi4py import MPI
import torch
from math import ceil
from random import Random
import torchvision
from torchvision import datasets, transforms

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chunks. """
    def __init__(self, data, sizes, seed=1234, isNonIID=True):
        self.data = data
        self.partitions = []

        if isNonIID:
            self.partitions = self.getNonIIDdata(data, sizes, seed)
        else:
            rng = Random()
            rng.seed(seed)
            data_len = len(data)
            indexes = [x for x in range(0, data_len)]
            for frac in sizes:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    #def worker_data(self, partition):
    #    return Partition(self.data, self.partitions[partition])

    def train_val_split(self, worker_partition, val_split):
        lengths = [int(len(worker_partition) * (1-val_split)), int(len(worker_partition) * val_split)]
        train_set, val_set = torch.utils.data.random_split(worker_partition, lengths)
        return Partition(self.data, train_set), Partition(self.data, val_set)

    def getNonIIDdata(self, data, sizes, rank, val_split=0.25, seed=1234):

        rng = Random()
        rng.seed(seed)

        # labelList = list()
        # for i in range(len(data)):
        #    labelList.append(labels[i])

        # Determine labels & create a dictionary storing all data point indices with their corresponding label
        # labelList = data.train_labels
        labelList = data.targets
        labelIdxDict = dict()
        for idx, label in enumerate(labelList):
            labelIdxDict.setdefault(label, [])
            labelIdxDict[label].append(idx)

        # Determine number of labels and create a list of these labels
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]

        # Create list of indices which point to most recent corresponding label in the data
        labelIdxPointer = [0] * labelNum

        # Create partition of the data for each worker
        partitions = [list() for _ in range(len(sizes))]
        eachPartitionLen = int(len(labelList)/len(sizes))
        # Determine the number of labels per worker (num partitions)
        majorLabelNumPerPartition = ceil(labelNum/len(partitions))

        basicLabelRatio = 0.4
        interval = 1
        labelPointer = 0

        # basic part
        # iterate through each of the partitions
        for partPointer in range(len(partitions)):
            # create a list of labels that will be predominant for a worker
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                # add the predominant labels to the list
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            # add in these predominant labels to the partition of the worker
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        # random part
        # construct a list of the remianing data points that haven't been added to a partition
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])

        # randomly shuffle the labels up so they are not in order by their label
        rng.shuffle(remainLabels)

        # iterate over the workers to add in random labels to their partition
        for partPointer in range(len(partitions)):
            # find the gap needed to be filled to meet the expected partition length (needed - what is there already)
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            # fill the partition to the desired length
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            # randomly shuffle the partition
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]

        # ANOTHER EDIT IS NEEDED SO ONE DOESNT NEED TO BUILD THE ENTIRE DICTIONARY, JUST ENOUGH FOR THE ONE WORKER
        # Before returning, Split into two partitions: 1 for training (75%) and one for validation (25%)
        worker_partition = partitions[rank]
        lengths = [int(len(worker_partition) * (1 - val_split)), int(len(worker_partition) * val_split)]
        train_set, val_set = torch.utils.data.random_split(worker_partition, lengths)

        return Partition(self.data, train_set), Partition(self.data, val_set)


def partition_dataset(rank, size, args):

    if args.downloadCifar == 1:
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = "c58f30108f718f92721af3b95e74349a"
        torchvision.datasets.utils.download_and_extract_archive(url, args.datasetRoot, filename=filename, md5=tgz_md5)
        MPI.COMM_WORLD.Barrier()

    if rank == 0:
        print('==> load train data')

    if args.dataset == 'cifar10':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        trainset = torchvision.datasets.CIFAR10(root=args.datasetRoot,
                                                train=True,
                                                download=True,
                                                transform=transform_train)

        partition_sizes = [1.0 / size for _ in range(size)]

        train_set, val_set = DataPartitioner(trainset, partition_sizes, isNonIID=True)

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.bs,
                                                   shuffle=True,
                                                   pin_memory=True)

        val_loader = torch.utils.data.DataLoader(val_set,
                                                   batch_size=args.bs,
                                                   shuffle=True,
                                                   pin_memory=True)

        MPI.COMM_WORLD.Barrier()

        if rank == 0:
            print('==> load test data')

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root=args.datasetRoot,
                                               train=False,
                                               download=True,
                                               transform=transform_test)

        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=64,
                                                  shuffle=False)
        MPI.COMM_WORLD.Barrier()

    return train_loader, test_loader, val_loader
