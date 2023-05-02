import numpy as np
import time
import torch
from Communicators.CommHelpers import flatten_tensors, unflatten_tensors


class decenCommunicator:
    """
    decentralized averaging according to a topology sequence
    For DSGD: Set i1 = 0 and i2 > 0 (any number it doesn't matter)
    For PD-SGD: Set i1 > 0 and i2 = 1
    For LD-SGD: Set i1 > 0 and i2 > 1
    """

    def __init__(self, rank, size, comm, topology, i1, i2):
        self.comm = comm
        self.rank = rank
        self.size = size
        self.topology = topology
        self.neighbor_list = self.topology.neighbor_list
        self.neighbor_weights = topology.neighbor_weights
        self.degree = len(self.neighbor_list)
        self.i1 = i1
        self.i2 = i2
        self.iter = 0
        self.comm_iter = 0

    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)

    def averaging(self):

        self.comm.Barrier()
        tic = time.time()

        # compute self weight according to degree
        selfweight = 1 - np.sum(self.neighbor_weights)
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(self.send_buffer, alpha=selfweight)

        send_buff = self.send_buffer.detach().numpy()
        self.recv_tmp = np.empty_like(send_buff)
        # decentralized averaging
        for idx, node in enumerate(self.neighbor_list):
            self.comm.Sendrecv(sendbuf=send_buff, source=node, recvbuf=self.recv_tmp, dest=node)
            # Aggregate neighbors' models: alpha * sum_j x_j
            self.recv_buffer.add_(torch.from_numpy(self.recv_tmp), alpha=self.neighbor_weights[idx])

        self.comm.Barrier()
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

        # Have to have this here because of the case that i1 = 0 (cant do 0 % 0)
        self.iter += 1
        comm_time = 0

        # I1: Number of Local Updates Communication Set
        if self.iter % (self.i1+1) == 0:

            self.comm_iter += 1
            # stack all model parameters into one tensor list
            self.tensor_list = list()
            for param in model.parameters():
                self.tensor_list.append(param)

            # necessary preprocess
            self.prepare_comm_buffer()

            # decentralized averaging according to activated topology
            # record the communication time
            comm_time += self.averaging()

            # update local models
            self.reset_model()

            # I2: Number of DSGD Communication Set
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1

        return comm_time
