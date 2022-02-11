import numpy as np
import time
from mpi4py import MPI
import torch
from comm_helpers import flatten_tensors, unflatten_tensors


class AsyncDecentralized:

    def __init__(self, rank, size, topology, sgd_updates, sgd_max):
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
        self.requests = [MPI.REQUEST_NULL for _ in range(10000)]
        self.requests2 = [MPI.REQUEST_NULL for _ in range(10000)]
        self.count = 0
        self.count2 = 0


        self.testAcc = -1.0 * np.ones(self.degree)
        self.sgd_updates = sgd_updates
        self.init_sgd_updates = sgd_updates
        self.sgd_max = sgd_max
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

    def personalize(self, test_acc):

        worker_acc = -1
        # Do something about this later...
        worker_buff = np.empty(3)

        tic = time.time()
        for idx, node in enumerate(self.neighbor_list):
                    count = 0
                    while True:
                        req2 = self.comm.Irecv(worker_buff, source=node, tag=node+self.size)
                        if not req2.Test():
                            if count == 0:
                                # If no messages available, keep unchanged
                                req2.Cancel()
                                break
                            else:
                                req2.Cancel()
                                self.testAcc[idx] = worker_acc
                                break

                        worker_acc = worker_buff[0]
                        count += 1

        toc = time.time()

        if not any(self.testAcc == -1.0):
            if test_acc <= np.min(self.testAcc) and self.sgd_updates < self.sgd_max:
                self.sgd_updates += 1
                print('Rank %d Had The Worst Accuracy at %f' % (self.rank, test_acc))
            elif test_acc > np.min(self.testAcc) and self.sgd_updates > self.init_sgd_updates:
                self.sgd_updates -= 1

        return toc - tic

    def averaging(self, model):

        # necessary preprocess
        self.prepare_send_buffer(model)
        self.avg_model = torch.zeros_like(self.send_buffer)

        worker_model = np.empty_like(self.avg_model)
        prev_model = np.empty_like(self.avg_model)

        tic = time.time()
        for idx, node in enumerate(self.neighbor_list):
                    count = 0
                    while True:
                        req = self.comm.Irecv(worker_model, source=node, tag=node)
                        if not req.Test():
                            if count == 0:
                                # If no messages available, take one's own model as the model to average
                                req.Cancel()
                                self.avg_model.add_(self.send_buffer, alpha=self.neighbor_weights[idx])
                                break
                            else:
                                req.Cancel()
                                self.avg_model.add_(torch.from_numpy(prev_model), alpha=self.neighbor_weights[idx])
                                # if any(np.isnan(prev_model)) or prev_model[-1] == 1:
                                #    print('Buffer Issue With Value %f When Updating From Rank %d' % (prev_model[-1], self.rank))
                                break
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

        # Preprocess
        self.prepare_send_buffer(model)
        send_buffer = self.send_buffer.detach().numpy()

        # Time
        tic = time.time()

        if self.count >= 10000-self.degree:
            self.count = 0

        for idx, node in enumerate(self.neighbor_list):
            self.requests[self.count] = self.comm.Isend(send_buffer, dest=node, tag=self.rank)
            self.count += 1

        toc = time.time()

        return toc - tic

    def communicate(self, model, test_acc, flag):

        self.iter += 1
        comm_time = 0

        if self.iter % self.sgd_updates == 0:
            comm_time += self.broadcast(model)
            comm_time += self.averaging(model)
            if flag:
                comm_time += self.personalize(test_acc)
        else:
            comm_time += self.broadcast(model)

        return comm_time

    def send_accuracy(self, test_acc):

        send_buff = test_acc*np.ones(3)
        if self.count2 >= 10000-self.degree:
            self.count2 = 0

        # Time
        tic = time.time()
        for node in self.neighbor_list:
            self.requests2[self.count2] = self.comm.Isend(send_buff, dest=node, tag=self.rank+self.size)
            self.count2 += 1
        toc = time.time()

        return toc - tic
