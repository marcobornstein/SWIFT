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
            if file.startswith(datatype+".log"):
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

    acc_data = unpack_data(path, epoch, 1, 'consensus-average')
    print(acc_data[0::10])


