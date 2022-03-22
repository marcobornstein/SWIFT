import numpy as np
import os
from mpi4py import MPI
import torch
from comm_helpers import flatten_tensors, unflatten_tensors
from Misc import test_accuracy
from DataPartition import consensus_test_data, consensus_train_data


def model_avg(worker_size, model, args):

    consensus_accuracy = list()
    model_diff = list()
    tensor_list = list()
    for param in model.parameters():
        tensor_list.append(param)
    send_buffer = flatten_tensors(tensor_list).cpu()

    test_data = consensus_test_data(args)
    train_data = consensus_train_data(5000, args)

    for epoch in range(args.epoch):

        # Clear the buffers
        avg_model = torch.zeros_like(send_buffer)
        worker_models = [np.empty_like(avg_model) for _ in range(worker_size)]
        np_avg_model = np.zeros_like(avg_model)

        # Get weighting (build a function) -- For now, make it uniform
        weighting = (1/worker_size) * np.ones(worker_size)

        for rank in range(worker_size):
            MPI.COMM_WORLD.Recv(worker_models[rank], source=rank, tag=rank+10*worker_size)
            avg_model.add_(torch.from_numpy(worker_models[rank]), alpha=weighting[rank])
            np_avg_model += worker_models[rank] * weighting[rank]

        # Reset local models to be the averaged model
        for f, t in zip(unflatten_tensors(avg_model.cuda(), tensor_list), tensor_list):
            with torch.no_grad():
                t.set_(f)

        # add in a forward pass to stabilize running mean/std
        model.train()
        for batch_idx, (data, target) in enumerate(train_data):
            data = data.cuda(non_blocking=True)
            model(data)

        # Compute accuracy for consensus model
        test_acc = test_accuracy(model, test_data)
        consensus_accuracy.append(test_acc)
        print('Consensus Accuracy for Epoch %d is %.3f' % (epoch, test_acc))

        for rank in range(worker_size):
            model_diff.append(np.linalg.norm(np_avg_model - worker_models[rank]))

    subfolder = args.outputFolder + '/' + args.name + '-' + str(args.graph) + '-' \
                + str(args.sgd_steps) + 'sgd-' + str(args.epoch) + 'epochs'

    isExist = os.path.exists(subfolder)
    if not isExist:
        os.makedirs(subfolder)

    np.savetxt(subfolder + '/consensus-average-' + args.comm_style + '.log', consensus_accuracy, delimiter=',')
    np.savetxt(subfolder + '/model-diff-' + args.comm_style + '.log', model_diff, delimiter=',')

    with open(subfolder + '/ExpDescription', 'w') as f:
        f.write(str(args) + '\n')
        f.write(args.description + '\n')
