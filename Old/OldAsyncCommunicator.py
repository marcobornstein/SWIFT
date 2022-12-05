import numpy as np
import time
from mpi4py import MPI
import torch
from comm_helpers import flatten_tensors, unflatten_tensors


class AsyncDecentralized:

    def __init__(self, rank, size, topology, sgd_updates):
        """ Initialize the Asynchronous Decentralized Communicator """

        # Graph initialization
        self.topology = topology
        self.neighbor_list = self.topology.neighbor_list
        self.neighbor_weights = topology.neighbor_weights
        self.degree = len(self.neighbor_list)

        # Initialize MPI variables
        self.comm = MPI.COMM_WORLD
        self.rank = rank
        self.size = size
        self.requests = [MPI.REQUEST_NULL for _ in range(self.degree)]

        self.sgd_updates = sgd_updates
        self.iter = 0

    def prepare_send_buffer(self, model):

        # stack all model parameters into one tensor list
        self.tensor_list = list()

        for param in model.parameters():
            self.tensor_list.append(param)

        # flatten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()

    def reset_model(self):
        # Reset local models to be the averaged model
        for f, t in zip(unflatten_tensors(self.avg_model.cuda(), self.tensor_list), self.tensor_list):
            with torch.no_grad():
                t.set_(f)

    def averaging(self, model):

        # necessary preprocess
        self.prepare_send_buffer(model)
        self.avg_model = torch.zeros_like(self.send_buffer)
        worker_model = np.empty_like(self.avg_model)

        tic = time.time()

        # make list of booleans and then put while loop outside of for loop. if the request is false then make boolean false, etc.
        '''
        flag = [False for _ in range(self.degree)]
        req = [MPI.REQUEST_NULL for _ in range(self.degree)]
        count = 0
        while not all(flag):
            for idx, node in enumerate(self.neighbor_list):
                req[idx] = self.comm.Irecv(worker_model, source=node, tag=node)


            for i in range(self.degree):
                if not req[i].Test():
                    req[i].Cancel()
                    flag[i] = True
            count +=1

        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        for idx, node in enumerate(self.neighbor_list):
            flag = True
            count = 0
            while flag:
                req = self.comm.Irecv(worker_model, source=node, tag=node)
                if not req.Test():
                    if count == 0:
                        # If no messages available, take one's own model as the model to average
                        worker_model = self.send_buffer.detach().numpy()
                    req.Cancel()
                    flag = False
                count += 1

            self.avg_model.add_(torch.from_numpy(worker_model), alpha=self.neighbor_weights[idx])

        '''

        # Im sending maybe an incorrect model, needs to be a previous model

        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        for idx, node in enumerate(self.neighbor_list):
            flag = True
            count = 0
            prev_model = np.empty_like(self.avg_model)
            while flag:
                req = self.comm.Irecv(worker_model, source=node, tag=node)
                if not req.Test():
                    if count == 0:
                        # print('Rank %d Received No Messages from Rank %d' % (self.rank, node))
                        # If no messages available, take one's own model as the model to average
                        req.Cancel()
                        self.avg_model.add_(self.send_buffer, alpha=self.neighbor_weights[idx])
                        flag = False
                    else:
                        # print('Rank %d Received %d Messages from Rank %d' % (self.rank, count, node))
                        req.Cancel()
                        self.avg_model.add_(torch.from_numpy(prev_model), alpha=self.neighbor_weights[idx])
                        flag = False
                prev_model = worker_model
                count += 1

        # compute self weight according to degree
        selfweight = 1 - np.sum(self.neighbor_weights)

        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.avg_model.add_(self.send_buffer, alpha=selfweight)

        toc = time.time()

        # update local models
        self.reset_model()

        return toc - tic

    def broadcast(self, model):

        # necessary preprocess
        self.prepare_send_buffer(model)

        tic = time.time()

        for idx, node in enumerate(self.neighbor_list):
            self.requests[idx] = self.comm.Isend(self.send_buffer.detach().numpy(), dest=node, tag=self.rank)

        toc = time.time()

        return toc - tic

    def communicate(self, model):

        self.iter += 1

        if self.iter % self.sgd_updates == 0:
            a = self.broadcast(model)
            b = self.averaging(model)
            comm_time = a + b
        else:
            comm_time = self.broadcast(model)

        return comm_time
