import matplotlib.pyplot as plt
import os
import numpy as np
import sys


def unpack_data(directory_path, epoch, num_workers, datatype):

    directory = os.path.join(directory_path)
    data = np.zeros((epoch, num_workers))

    for root, dirs, files in os.walk(directory):
        j = 0
        for file in files:
            if file.startswith(datatype):
                f = open(directory_path+'/'+file, 'r')
                i = 0
                for line in f:
                    data[i, j] = line
                    i += 1
                j += 1

    return data


def unpack_data2(directory_path, epoch, num_workers, datatype):

    directory = os.path.join(directory_path)
    data = np.zeros((epoch, num_workers))

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(datatype):
                j = int(file[6])
                f = open(directory_path+'/'+file, 'r')
                i = 0
                for line in f:
                    data[i, j] = line
                    i += 1

    return data


def unpack_data3(directory_path, epoch, num_workers, datatype, communicator):

    directory = os.path.join(directory_path)
    data = np.zeros((6, num_workers))
    count = 0

    for root, dirs, files in os.walk(directory):

        for dir in dirs:
            if dir.startswith(communicator):
                count = int(dir[-11]) - 1
                new_directory_path = directory_path + '/' + dir
                new_directory = os.path.join(new_directory_path)
                temp_data = np.zeros((epoch, num_workers))
                for root, dirs, files in os.walk(new_directory):
                    for file in files:
                        if file.endswith(datatype):
                            j = int(file[1])
                            f = open(new_directory_path + '/' + file, 'r')
                            i = 0
                            for line in f:
                                temp_data[i, j] = line
                                i += 1
                # cum_temp_data = np.cumsum(temp_data)
                # data[count, :] = cum_temp_data[-1, :]
                data[count, :] = np.sum(temp_data, axis=0)
                count += 1
        data[-1, :] = np.average(data[:-1, :], axis=0)
    return data


if __name__ == "__main__":

    args = sys.argv

    if len(args) != 5:
        raise ValueError('There should be 4 arguments!')

    path = args[1]
    epoch = int(args[2])
    num_workers = int(args[3])
    comm = args[4]

    time_data = unpack_data3(path, epoch, num_workers, 'total-time.log', comm)
    print(time_data)

    # acc_data = unpack_data(path, epoch, 1, 'consensus-average')
    # print(acc_data[0])
    # print(acc_data[9::10])

    '''
    time_data = unpack_data2(path, epoch, num_workers, 'total-time.log')
    cum_time_data = np.cumsum(time_data, axis=0)
    print('Total Time Until Completion for each Worker:')
    print(cum_time_data[-1, :])
    print('Average Time per Epoch for each Worker:')
    print(cum_time_data[-1, :]/epoch)

    time_data2 = unpack_data2(path, epoch, num_workers, 'epoch-time.log')
    avg_time_data = np.average(time_data, axis=0)
    min_time_data = np.min(time_data, axis=0)
    print('Average Epoch Time For Each Worker:')
    print(avg_time_data)
    print('Minimum Epoch Time For Each Worker:')
    print(min_time_data)
    # max_time_data = np.max(time_data, axis=0)
    # print('Maximum Epoch Time For Each Worker:')
    # print(max_time_data)
    print('==================================================================')
    '''
