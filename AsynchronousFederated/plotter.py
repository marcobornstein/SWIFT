import matplotlib.pyplot as plt
import os
import numpy as np
import sys


def unpack_data(directory_path, epoch, num_workers, datatype):
    directory = os.path.join(directory_path)
    data = np.zeros((epoch, num_workers))

    if datatype == 'time':
        for root, dirs, files in os.walk(directory):
            j = 0
            for file in files:
                if file.endswith(datatype + ".log"):
                    f = open(directory_path + '/' + file, 'r')
                    i = 0
                    time_sum = 0
                    for line in f:
                        time_sum += float(line)
                        data[i, j] = time_sum
                        i += 1
                    j += 1
    else:
        for root, dirs, files in os.walk(directory):
            j = 0
            for file in files:
                if file.endswith(datatype+".log"):
                    f = open(directory_path+'/'+file, 'r')
                    i = 0
                    for line in f:
                        data[i, j] = line
                        i += 1
                    j += 1

    return data


if __name__ == "__main__":

    args = sys.argv

    if len(args) != 5:
        raise ValueError('There should be 4 arguments!')

    path = args[1]
    epoch = int(args[2])
    num_workers = int(args[3])

    # datatypes = ['tacc', 'acc', 'losses', 'time', 'comptime', 'commtime']
    datatypes = ['tacc', 'acc', 'losses']
    ylabels = ['Training-Accuracy', 'Test-Accuracy', 'Training-Loss']

    output_folder = './Figures/'
    output_name = args[4]

    time_data = unpack_data(path, epoch, num_workers, 'time')

    for i in range(len(datatypes)):
        
        data = unpack_data(path, epoch, num_workers, datatypes[i])
        
        fig = plt.figure()
        for j in range(num_workers):
            plt.plot(range(1, epoch+1), data[:, j])
            plt.xlabel('Epochs')
            plt.ylabel(ylabels[i])

        plt.savefig(output_folder+output_name+'-'+ylabels[i]+'.png')

        for j in range(num_workers):
            plt.plot(range(1, epoch+1), time_data[:, j])
            plt.xlabel('Wall Time (Seconds)')
            plt.ylabel(ylabels[i])

        plt.savefig(output_folder+output_name+'-WallTime-'+ylabels[i]+'.png')
