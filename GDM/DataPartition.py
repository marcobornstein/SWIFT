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
            self.partitions, self.val = self.getNonIIDdata(rank, data, sizes, 0,
                                                            val_split=val_split, seed=seed)


    def train_val_split(self):
        return Partition(self.data, self.partitions), Partition(self.data, self.val)

    def getNonIIDdata(self, rank, data, partition_sizes, degree_noniid, val_split=0.25, seed=1234):
        # Note: method may assign same data points to different workers if seed changes
        # Is called once per each rank (albeit unnecessarily)

        if degree_noniid < 0 or degree_noniid > 1:
            print("Warning: clipping degree_noniid to [0, 1]")
        degree_noniid = max(0, min(1, degree_noniid))

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

        # The TOTAL number of datapoints in each worker
        total_worker_sizes = []
        # The below code partitions num_data into sections, such that all sections sum to num_data
        # e.g.: [1/2, 1/2] partition and num_data=3 would make total_worker_sizes [2, 1]
        total_data_worker = 0
        for worker in range(num_workers):
            total_worker_sizes.append(int(partition_sizes[worker] * num_data))
            total_data_worker += total_worker_sizes[-1]

        # Increments by 1 to account for int truncation
        rem = num_data - total_data_worker
        while rem != 0:
            assert rem > 0 and rem < num_workers
            rem -= 1
            total_worker_sizes[rem] += 1

        # Splits each worker's datapoints into noniid and iid data
        # Represented by a tuple (allocated noniid, allocated iid)
        worker_split = []
        non_iid_sum = 0
        for worker in range(num_workers):
            non_iid_size = ceil(total_worker_sizes[worker] * degree_noniid)
            non_iid_sum += non_iid_size
            worker_split.append((non_iid_size, total_worker_sizes[worker] - non_iid_size))
    
        # Determines how many noniid samples from each label to use
        # Should be # of samples in a label * degree_noniid, but need to account for remainder/int truncation
        # Which is done below
        total_label_niids = []
        total_data_niid = 0
        for label in labelIdxDict:
            total_label_niids.append(int(len(labelIdxDict[label]) * degree_noniid))
            total_data_niid += total_label_niids[-1]
        
        # Adds or subtracts one to make everything sum to the total number of allocated noniid data across all workers
        rem = non_iid_sum - total_data_niid
        while rem != 0:
            assert abs(rem) < num_labels
            if rem > 0:
                rem -= 1
                total_label_niids[rem] += 1
            else:
                total_label_niids[rem] -= 1
                rem += 1
        
        # Selects which indices to use for iid data
        iid_inds = []
        for label in labelIdxDict:
            rng.shuffle(labelIdxDict[label])
            
            iid_inds.extend(labelIdxDict[label][total_label_niids[label]:])
            labelIdxDict[label] = labelIdxDict[label][:total_label_niids[label]]

        # data_inds stores noniid data indices, will select from it later for current worker
        data_inds = []
        curr_bin = 0

        # Determines which segment of iid data to select from
        # Each worker/rank has a disjoint segment over the iid data
        running_iid = 0
        iid_start_ind = 0
        iid_end_ind = 0

        for worker_idx in range(num_workers):            
            if worker_idx == rank:
                iid_start_ind = running_iid
                iid_end_ind = running_iid + worker_split[worker_idx][1]

            running_iid += worker_split[worker_idx][1]

            # Fills up worker with to_fill noniid data
            to_fill = worker_split[worker_idx][0]
            while to_fill > 0:
                num_in_bin = len(labelIdxDict[curr_bin])
                # How much to take from the bin/label
                take_num = min(to_fill, num_in_bin)
                to_fill -= take_num

                # Transfers data over
                take_ind = num_in_bin - take_num
                data_inds.extend(labelIdxDict[curr_bin][take_ind:])
                labelIdxDict[curr_bin] = labelIdxDict[curr_bin][:take_ind]

                # Move to next label
                curr_bin += 1
                curr_bin %= num_labels
            
            # No need to keep on going, only return data for worker rank
            if worker_idx == rank:
                break

        # Collects noniid data
        worker_data = data_inds[(-worker_split[rank][0]):]
        # Collects iid data
        rng.shuffle(iid_inds)
        worker_data.extend(iid_inds[iid_start_ind:iid_end_ind])           

        # Before returning, Split into two partitions: 1 for training (75%) and one for validation (25%)
        valid_len = int(total_worker_sizes[rank] * val_split)
        lens = [total_worker_sizes[rank] - valid_len, valid_len]
        train_set, val_set = torch.utils.data.random_split(worker_data, lens)
        return train_set, val_set


def partition_dataset(rank, size, comm, val_split, args):

    # if no dataset file specified, create new 'Data' folder
    if args.datasetRoot == None:
        if not os.path.isdir('Data'):
            os.mkdir('Data')
        datasetRoot = 'Data'
        downloadCifar = 1
    else:
        datasetRoot = args.datasetRoot
        downloadCifar = args.downloadCifar

    if downloadCifar == 1:
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = "c58f30108f718f92721af3b95e74349a"
        torchvision.datasets.utils.download_and_extract_archive(url, datasetRoot, filename=filename, md5=tgz_md5)
        comm.Barrier()

    if rank == 0:
        print('==> load train data')

    if args.dataset == 'cifar10':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        trainset = torchvision.datasets.CIFAR10(root=datasetRoot,
                                                train=True,
                                                download=True,
                                                transform=transform_train)

        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, rank, degree_noniid=args.degree_noniid,
                                    val_split=val_split, isNonIID=args.noniid)
        train_set, val_set = partition.train_val_split()

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.bs,
                                                   shuffle=True,
                                                   pin_memory=True)

        #val_loader = torch.utils.data.DataLoader(val_set,
        #                                           batch_size=args.bs,
        #                                           shuffle=True,
        #                                           pin_memory=True)

        comm.Barrier()
        if rank == 0:
            print('==> load test data')

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root=datasetRoot, train=False, download=True,
                                               transform=transform_test)

        t1, t2 = torch.utils.data.random_split(testset, [500, 9500])

        test_loader = torch.utils.data.DataLoader(t1, batch_size=64, shuffle=False)
        comm.Barrier()

    return train_loader, test_loader #, val_loader


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

    indexes = [x for x in range(0, train_size)]
    train_set = Partition(trainset, indexes)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.bs,
                                               shuffle=True,
                                               pin_memory=True)

    return train_loader
