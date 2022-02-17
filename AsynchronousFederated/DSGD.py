import numpy as np
import time
import torch
from mpi4py import MPI
from comm_helpers import flatten_tensors, unflatten_tensors


class decenCommunicator:
    """ decentralized averaging according to a topology sequence """

    def __init__(self, rank, size, topology):
        self.comm = MPI.COMM_WORLD
        self.rank = rank
        self.size = size
        self.topology = topology
        self.neighbor_list = self.topology.neighbor_list
        self.neighbor_weights = topology.neighbor_weights
        self.degree = len(self.neighbor_list)

    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)

    def averaging(self):

        self.comm.barrier()
        tic = time.time()

        # decentralized averaging
        for idx, node in enumerate(self.neighbor_list):
            self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=node, dest=node)
            # Aggregate neighbors' models: alpha * sum_j x_j
            # self.recv_buffer.add_(self.neighbor_weight, self.recv_tmp)
            self.recv_buffer.add_(self.recv_tmp, alpha=self.neighbor_weights[idx])

        # compute self weight according to degree
        selfweight = 1 - np.sum(self.neighbor_weights)
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(self.send_buffer, alpha=selfweight)

        self.comm.barrier()
        toc = time.time()

        return toc - tic

    def reset_model(self):
        # Reset local models to be the averaged model
        for f, t in zip(unflatten_tensors(
                self.recv_buffer.cuda(), self.tensor_list),
                self.tensor_list):
            with torch.no_grad():
                t.set_(f)

    def communicate(self, model):

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            # self.tensor_list.append(param.data)
            self.tensor_list.append(param)

            # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time = self.averaging()

        # update local models
        self.reset_model()

        return comm_time