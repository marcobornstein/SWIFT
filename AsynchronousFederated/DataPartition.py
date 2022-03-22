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
    def __init__(self, data, sizes, rank, seed=1234, degree_noniid=0.7, isNonIID=True, val_split=0.25):
        self.data = data

        if isNonIID:
            self.partitions, self.val = self.getNonIIDdata(rank, data, sizes, degree_noniid,
                                                           val_split=val_split, seed=seed)
        else:
            partitions = list()
            rng = Random()
            rng.seed(seed)
            data_len = len(data)
            indexes = [x for x in range(0, data_len)]
            # rng.shuffle(indexes)
            for frac in sizes:
                part_len = int(frac * data_len)
                partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]
            worker_data_len = len(partitions[rank])
            self.val = partitions[rank][0:int(val_split*worker_data_len)]
            self.partitions = partitions[rank][int(val_split*worker_data_len):]

    def train_val_split(self):
        return Partition(self.data, self.partitions), Partition(self.data, self.val)

    def getNonIIDdata(self, rank, data, partition_sizes, degree_noniid, val_split=0.25, seed=1234):

        rng = Random()
        rng.seed(seed)
        num_workers = len(partition_sizes)

        # Determine labels & create a dictionary storing all data point indices with their corresponding label
        labelList = data.targets
        num_data = len(labelList)
        labelIdxDict = dict()
        for idx, label in enumerate(labelList):
            labelIdxDict.setdefault(label, [])
            labelIdxDict[label].append(idx)

        # Determine number of labels and create a list of these labels
        num_labels = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]

        # Create list of indices which point to most recent corresponding label in the data
        labelIdxPointer = [0] * num_labels

        # Initialize data partition list for each worker
        partitions = [list() for _ in range(num_workers)]

        # Determine the number of labels unique to each
        labels_per_worker = ceil(num_labels / num_workers)
        for worker_idx in range(num_workers):

            # Determine partition size and amount of non-iid data needed to fill
            partition_size = partition_sizes[worker_idx] * num_data
            desired_non_iid_data_len = int(degree_noniid * partition_size)

            # Determine the single label designated to the worker
            start_idx = worker_idx * labels_per_worker
            label_list = [(start_idx + i) % num_labels for i in range(labels_per_worker)]

            per_label_size = [int(desired_non_iid_data_len / labels_per_worker) for _ in range(labels_per_worker-1)]
            per_label_size.append(desired_non_iid_data_len - sum(per_label_size))

            for idx, label in enumerate(label_list):

                # Set the amount of data still needed to fill
                needed_data_len = per_label_size[idx]

                # until designated partition is filled with allotted non-iid data:
                while needed_data_len > 0:

                    # Determine the dictionary key corresponding to the assigned label
                    key = labelNameList[label]

                    # Determine the current number of data remaining for the given label
                    start = labelIdxPointer[label]
                    remaining_data = len(labelIdxDict[key][start:])

                    # If enough non-iid data is available to take, take it all
                    if needed_data_len < remaining_data:
                        partitions[worker_idx].extend(labelIdxDict[key][start:needed_data_len])
                        labelIdxPointer[label] += needed_data_len
                        needed_data_len = 0
                    # Else, take the rest of the available data and move to the next label and continue this process
                    else:
                        partitions[worker_idx].extend(labelIdxDict[key][start:])
                        labelIdxPointer[label] = len(labelIdxDict[key])
                        needed_data_len -= remaining_data
                        label += 1

        # fill the rest of the partition with random iid data if there's room left
        # construct a list of the remaining data points that haven't been added to a partition
        remainLabels = list()
        for labelIdx in range(num_labels):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])

        # randomly shuffle the labels up so they are not in order by their label
        rng.shuffle(remainLabels)

        # iterate over the workers to add in random labels to their partition
        for worker_idx in range(num_workers):
            # Find designated partition size
            partition_size = partition_sizes[worker_idx] * num_data
            # find the gap needed to be filled to meet the expected partition length (needed - what is there already)
            missing_data_len = int(partition_size - len(partitions[worker_idx]))
            # fill the partition to the desired length
            partitions[worker_idx].extend(remainLabels[:missing_data_len])
            # randomly shuffle the partition
            rng.shuffle(partitions[worker_idx])
            remainLabels = remainLabels[missing_data_len:]

        # Before returning, Split into two partitions: 1 for training (75%) and one for validation (25%)
        worker_partition = partitions[rank]
        worker_len = len(worker_partition)
        rem = worker_len - (int(worker_len * (1 - val_split)) + int(worker_len * val_split))
        lengths = [int(worker_len * (1 - val_split)) + rem, int(worker_len * val_split)]
        train_set, val_set = torch.utils.data.random_split(worker_partition, lengths)

        return train_set, val_set


def partition_dataset(rank, size, comm, args):

    if args.downloadCifar == 1:
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = "c58f30108f718f92721af3b95e74349a"
        torchvision.datasets.utils.download_and_extract_archive(url, args.datasetRoot, filename=filename, md5=tgz_md5)
        comm.Barrier()

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

        partition = DataPartitioner(trainset, partition_sizes, rank, args.degree_noniid,
                                    val_split=0.25, isNonIID=args.noniid)
        train_set, val_set = partition.train_val_split()


        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.bs,
                                                   shuffle=True,
                                                   pin_memory=True)

        val_loader = torch.utils.data.DataLoader(val_set,
                                                   batch_size=args.bs,
                                                   shuffle=True,
                                                   pin_memory=True)

        comm.Barrier()

        if rank == 0:
            print('==> load test data')

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root=args.datasetRoot, train=False, download=True,
                                               transform=transform_test)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
        comm.Barrier()

    return train_loader, test_loader, val_loader


def consensus_test_data(args):

    if args.downloadCifar == 1:
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = "c58f30108f718f92721af3b95e74349a"
        torchvision.datasets.utils.download_and_extract_archive(url, args.datasetRoot, filename=filename, md5=tgz_md5)
        MPI.COMM_WORLD.Barrier()

    if args.dataset == 'cifar10':

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root=args.datasetRoot,
                                               train=False,
                                               download=True,
                                               transform=transform_test)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return test_loader


def consensus_train_data(train_size, args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root=args.datasetRoot,
                                            train=True,
                                            download=True,
                                            transform=transform_train)

    partitions = list()
    rng = Random()
    rng.seed(1234)
    data_len = len(trainset)
    indexes = [x for x in range(0, data_len)]
    rng.shuffle(indexes)
    partition = indexes[0:train_size]
    train_set = Partition(trainset, partition)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.bs,
                                               shuffle=True,
                                               pin_memory=True)

    return train_loader
