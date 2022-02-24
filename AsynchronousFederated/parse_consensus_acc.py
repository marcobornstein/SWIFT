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
                print(file[8])
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
        j = 0
        for file in files:
            if file.endswith(datatype):
                f = open(directory_path+'/'+file, 'r')
                i = 0
                for line in f:
                    data[i, j] = line
                    i += 1
                j += 1

    return data


if __name__ == "__main__":

    args = sys.argv

    if len(args) != 4:
        raise ValueError('There should be 3 arguments!')

    path = args[1]
    epoch = int(args[2])
    num_workers = int(args[3])

    # acc_data = unpack_data(path, epoch, 1, 'consensus-average')
    # print(acc_data[0])
    # print(acc_data[9::10])

    time_data = unpack_data2(path, epoch, num_workers, 'epoch-time.log')
    cum_time_data = np.cumsum(time_data, axis=0)
    # print(cum_time_data[-1, :])
    avg_time_data = np.average(time_data, axis=0)
    min_time_data = np.min(time_data, axis=0)
    max_time_data = np.max(time_data, axis=0)

    print('Average Epoch Time For Each Worker:')
    print(avg_time_data)
    print('Minimum Epoch Time For Each Worker:')
    print(min_time_data)
    print('Maximum Epoch Time For Each Worker:')
    print(max_time_data)


