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
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 0

    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)

    def averaging(self, active_flags):

        self.comm.barrier()
        tic = time.time()

        # decentralized averaging
        degree = 0  # record the degree of each node
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != -1:
                    degree += 1
                    neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                    # Receive neighbor's model: x_j
                    self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest=neighbor_rank)
                    # Aggregate neighbors' models: alpha * sum_j x_j
                    # self.recv_buffer.add_(self.neighbor_weight, self.recv_tmp)
                    self.recv_buffer.add_(self.recv_tmp, alpha=self.neighbor_weight)

        # compute self weight according to degree
        selfweight = 1 - degree * self.neighbor_weight
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
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1

        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            # self.tensor_list.append(param.data)
            self.tensor_list.append(param)

            # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time = self.averaging(active_flags)

        # update local models
        self.reset_model()

        return comm_time