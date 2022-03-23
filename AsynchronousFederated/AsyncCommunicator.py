import numpy as np
import time
from mpi4py import MPI
import torch
from comm_helpers import flatten_tensors, unflatten_tensors

class AsyncDecentralized:

    def __init__(self, rank, size, comm, topology, sgd_updates, sgd_max):
        """ Initialize the Asynchronous Decentralized Communicator """

        # Graph initialization
        self.topology = topology
        self.neighbor_list = self.topology.neighbor_list
        self.neighbor_weights = topology.neighbor_weights
        self.degree = len(self.neighbor_list)

        # Initialize MPI variables
        self.comm = comm
        self.rank = rank
        self.size = size
        self.requests = [MPI.REQUEST_NULL for _ in range(10000)]
        self.requests2 = [MPI.REQUEST_NULL for _ in range(10000)]
        self.requests3 = [MPI.REQUEST_NULL for _ in range(self.degree)]
        self.requests4 = [MPI.REQUEST_NULL for _ in range(self.degree)]
        self.count = 0
        self.count2 = 0
        self.missed_msg = 0
        self.reqCount = 0
        self.reqCount2 = 0

        self.epochs = -1.0 * np.ones(self.degree)
        self.valAcc = -1.0 * np.ones(self.degree)
        self.exit = -1.0 * np.ones(self.degree)
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

    def personalize(self, epoch, val_acc, iidFlag):

        send_buff = np.empty(2)
        send_buff[0] = epoch
        send_buff[1] = val_acc

        if self.count2 >= 10000 - self.degree:
            self.count2 = 0
        if self.reqCount2 >= 10000 - self.degree:
            self.reqCount2 = 0

        # Time the send
        tic = time.time()
        for node in self.neighbor_list:
            self.requests2[self.count2] = self.comm.Isend(send_buff, dest=node, tag=self.rank + self.size)
            self.count2 += 1
            if self.requests2[self.reqCount2].Test():
                self.requests2[self.reqCount2].Wait()
                self.reqCount2 += 1
        toc = time.time()
        send_time = toc-tic

        worker_epoch = -1
        worker_vacc = -1
        worker_buff = np.empty(2)
        recv_nodes = list()

        tic = time.time()

        for idx, node in enumerate(self.neighbor_list):
            if self.comm.Iprobe(source=node, tag=node + self.size):
                recv_nodes.append((idx, node))

        for idx, node in recv_nodes:
            while True:
                req = self.comm.Irecv(worker_buff, source=node, tag=node + self.size)
                if not req.Test():
                    req.Cancel()
                    req.Free()
                    self.epochs[idx] = worker_epoch
                    self.valAcc[idx] = worker_vacc
                    break
                worker_epoch = worker_buff[0]
                worker_vacc = worker_buff[1]

        toc = time.time()
        recv_time = toc-tic

        if not any(self.epochs == -1.0) and not iidFlag:
            b = np.append(self.epochs, epoch)
            b = b / np.sum(b)
            self.neighbor_weights = b[:-1]

        if not any(self.valAcc == -1.0):
            if val_acc <= np.min(self.valAcc) and self.sgd_updates < self.sgd_max:
                self.sgd_updates += 1
                print('Rank %d Had The Worst Validation Accuracy at %f ' % (self.rank, val_acc))
            elif val_acc > np.min(self.valAcc) and self.sgd_updates > self.init_sgd_updates:
                self.sgd_updates -= 1

        return send_time+recv_time

    def averaging(self, model):

        # necessary preprocess
        self.prepare_send_buffer(model)
        self.avg_model = torch.zeros_like(self.send_buffer)
        worker_model = np.empty_like(self.avg_model)
        prev_model = np.empty_like(self.avg_model)
        recv_nodes = list()

        tic = time.time()

        for idx, node in enumerate(self.neighbor_list):
            if self.comm.Iprobe(source=node, tag=node):
                recv_nodes.append((idx, node))

        # weight_boost = (len(recv_nodes)+1) / (self.degree + 1)

        weight_sum = 0
        for idx, node in recv_nodes:
            weight = self.neighbor_weights[idx]
            weight_sum += weight
            while True:
                req = self.comm.Irecv(worker_model, source=node, tag=node)
                if not req.Test():
                    req.Cancel()
                    req.Free()
                    # self.avg_model.add_(torch.from_numpy(prev_model), alpha=self.neighbor_weights[idx]/weight_boost)
                    self.avg_model.add_(torch.from_numpy(prev_model), alpha=weight)
                    break
                prev_model = worker_model

        # compute self weight according to degree
        # selfweight = (1 - np.sum(self.neighbor_weights))/weight_boost
        selfweight = 1 - weight_sum

        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.avg_model.add_(self.send_buffer, alpha=selfweight)

        toc = time.time()

        # update local models
        self.reset_model()

        # determine the number of messages missed from other workers
        self.missed_msg += self.degree - len(recv_nodes)

        return toc - tic

    def broadcast(self, model):

        # Preprocess
        self.prepare_send_buffer(model)
        send_buffer = self.send_buffer.detach().numpy()

        if self.count >= 10000-self.degree:
            self.count = 0
        if self.reqCount >= 10000 - self.degree:
            self.reqCount = 0

        # Time
        tic = time.time()

        for idx, node in enumerate(self.neighbor_list):
            self.requests[self.count] = self.comm.Isend(send_buffer, dest=node, tag=self.rank)
            self.count += 1
            if self.requests[self.reqCount].Test():
                self.requests[self.reqCount].Wait()
                self.reqCount += 1

        toc = time.time()

        return toc - tic

    def communicate(self, model):

        self.iter += 1
        comm_time = 0

        if self.iter % self.sgd_updates == 0:
            comm_time += self.broadcast(model)
            comm_time += self.averaging(model)
        else:
            comm_time += self.broadcast(model)

        return comm_time

    def wait(self, model):

        buf = [np.empty(1) for _ in range(self.degree)]
        # Send out exit flag
        for idx, node in enumerate(self.neighbor_list):
            self.requests3[idx] = self.comm.Isend(np.ones(1), dest=node, tag=self.rank + 2*self.size)
            self.requests4[idx] = self.comm.Irecv(buf[idx], source=node, tag=node + 2 * self.size)

        # Preprocess
        self.prepare_send_buffer(model)
        send_buffer = self.send_buffer.detach().numpy()

        while any(self.exit == -1.0):
            if self.count >= 10000 - self.degree:
                self.count = 0
            if self.reqCount >= 10000 - self.degree:
                self.reqCount = 0
            for idx, node in enumerate(self.neighbor_list):
                count = 0
                while True:
                    if not self.requests4[idx].Test() or self.exit[idx] != -1.0:
                        if count == 0 and self.exit[idx] == -1.0:
                            self.requests[self.count] = self.comm.Isend(send_buffer, dest=node, tag=self.rank)
                            self.count += 1
                            if self.requests[self.reqCount].Test():
                                self.requests[self.reqCount].Wait()
                                self.reqCount += 1
                        break
                    self.exit[idx] = buf[idx]
                    count += 1
            time.sleep(0.5)

        print('Rank %d Had %d Missed Messages' % (self.rank, self.missed_msg))
