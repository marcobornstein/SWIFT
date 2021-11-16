import matplotlib.pyplot as plt
import os
import numpy as np
import sys


def plot_accuracy(directory_path, epoch, num_workers):
    directory = os.path.join(directory_path)
    train_accuracies = np.zeros((epoch, num_workers))
    for root, dirs, files in os.walk(directory):
        j = 0
        for file in files:
            if file.endswith("tacc.log"):
                f = open(directory_path+'/'+file, 'r')
                i = 0
                for line in f:
                    train_accuracies[i,j] = line
                    i += 1
                j += 1
    return train_accuracies

if __name__ == "__main__":

    args = sys.argv

    if len(args) != 5:
        raise ValueError('There should be 3 arguments!')

    path = args[1]
    epoch = int(args[2])
    num_workers = int(args[3])

    train_accuracy = plot_accuracy(path, epoch, num_workers)
        
    fig = plt.figure()
    for i in range(num_workers):
        plt.plot(train_accuracy[:, i], range(1, epoch+1))

    output_folder = './Figures/'
    output_name = args[4]

    plt.savefig(output_folder+output_name+'.png')
