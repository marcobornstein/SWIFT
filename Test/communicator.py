import numpy as np
import time
from mpi4py import MPI


class AsyncCentralServer:
    """ Class constructed to turn one process into a proxy asynchronous central server """

    def __init__(self, rank, size, x, function, ft, tau, T):
        """ Initialize the Central Server """

        # Initialize MPI variables
        self.comm = MPI.COMM_WORLD
        self.rank = rank
        self.size = size
        self.requests = [MPI.REQUEST_NULL for _ in range(0, size - 1)]

        # Initialize list which stores the corresponding index (0:size-1) to destination relationship
        self.destination = []

        # Store initial solution and objective function, initialize array storing iterate differences and obj. values
        self.x = x
        self.function = function
        self.x_diff = np.empty(0)
        self.obj = [self.function(self.x)]

        # Store SE-ACGD parameters
        self.H = self.function(self.x)
        self.diffH = 0
        self.L = 2
        self.F = ft
        self.tau = tau
        self.T = T

        # Initialize time storing
        self.times = [0.0]
        self.initial_time = 0

        # Initialize buffer for receiving MPI messages
        self.buffer = np.zeros(self.x.shape)

        # Store all the indices each worker will be working on in a list of a list
        self.worker_idx = []
        self.per_process = int(self.x.shape[0] / (self.size - 1))
        for i in range(self.size-1):
            if i != self.size-2:
                self.worker_idx.append(range(i*self.per_process, (i+1)*self.per_process))
            else:
                self.worker_idx.append(range(i*self.per_process, x.shape[0]))

    def send(self, msg, destination):
        """ Send (non-blocking) the next iterate to worker which just finished its work """

        self.comm.isend(msg, dest=destination, tag=self.rank)

    def hamiltonian(self):
        """ Compute the Hamiltonian as defined in the paper """

        return self.function(self.x) + \
               (self.L/(2*np.sqrt(self.tau)))*np.sum(np.arange(1, self.tau+1)*self.x_diff[-self.tau:])

    def flush(self):
        for k in range(40):
            for i in range(0, self.size):
                if i != self.rank:
                    if self.comm.iprobe(source=i, tag=i):
                        self.comm.recv(source=i, tag=i)

    def initial_request(self):
        """ Initialize Communication between Server and Workers """

        j = 0
        for i in range(0, self.size):
            if i != self.rank:
                # Send initial communication requests to each worker
                self.requests[j] = self.comm.irecv(source=i, tag=i)
                # Correlate the index and worker ID
                self.destination.append(i)
                j += 1

    def receive(self, num_msg):
        """ Receive Worker updates for a specified number of updates """

        j = 0
        msg_count = 0

        while msg_count < num_msg:
            if self.requests[j].Get_status():

                # Determine which part of the solution the updating worker works on
                idx = self.worker_idx[j]

                # Retrieve the message sent from the worker
                self.buffer[idx] = self.requests[j].wait()

                # Determine the worker ID to send back information to
                destination = self.destination[j]

                # Put out another communication request for the worker to keep performing computations
                self.requests[j] = self.comm.irecv(source=destination, tag=destination)

                # Update the solution and gradient/objective metrics
                self.x_diff = np.append(self.x_diff, np.linalg.norm(self.buffer[idx]))
                self.x[idx] += self.buffer[idx]
                self.obj.append(self.function(self.x))

                # Send back the new solution to the worker
                self.send(self.x, destination)

                # Complete the iteration and time it
                msg_count += 1
                self.times.append(time.time() - self.initial_time)
            j = (j+1) % (self.size-1)

        # Compute the new Hamiltonian and difference between Hamiltonians
        newH = self.hamiltonian()
        self.diffH = self.H - newH
        self.H = newH

        # If Hamiltonian decreases sufficiently return a "True" flag, if not then return a "False" flag
        if self.diffH > self.F:
            flag = True
        else:
            flag = False

        # Send the flags to each worker and cancel all older communication requests
        for i in range(0, self.size-1):
            self.comm.isend(flag, dest=self.destination[i], tag=self.destination[i])
            self.requests[i].cancel()

        return flag

    def large_gradient(self):
        """ Large-Gradient sub-algorithm (LG-ACGD) in paper (tau iterations) """

        flag = True
        while flag:
            self.initial_request()
            flag = self.receive(self.tau)
        return

    def perturb(self):
        """ Perturb sub-algorithm (P-ACGD) in paper (T iterations) """

        # Compute perturbation
        unit = np.random.normal(size=self.x.shape)
        unit = unit / np.linalg.norm(unit)
        r = np.random.uniform(low=0, high=1)
        r = r**(1 / unit.size)
        scale = 0.05
        pert = r * unit * scale

        # Perturb solution
        self.x += pert

        # Compute T ACGD updates
        self.initial_request()
        flag = self.receive(self.T)

        return flag

    def acgd(self):
        """ SE-ACGD Algorithm """

        # While the perturbation algorithm hasn't yet concluded
        flag = True
        self.initial_time = time.time()
        while flag:
            # Run LG-ACGD
            self.large_gradient()
            print('Stationary Point Reached, Need Perturbation')
            print('Current Objective Value: %f' % self.function(self.x))
            print('Current Hamiltonian Value: %f' % self.H)
            print('Current Hamiltonian Difference: %f' % self.diffH)
            print('===================================================')
            # Run P-ACGD
            flag = self.perturb()
            print('Post Perturbation:')
            print('Current Objective Value: %f' % self.function(self.x))
            print('Current Hamiltonian Value: %f' % self.H)
            print('Current Hamiltonian Difference: %f' % self.diffH)
            print('===================================================')
        return self.x, self.x_diff, np.array(self.obj), np.array(self.times)


class SyncCentralServer:
    """ Class constructed to turn one process into a proxy synchronous central server """

    def __init__(self, rank, size, x, function, ft, tau, T):
        """ Initialize the Central Server """

        # Initialize MPI variables
        self.comm = MPI.COMM_WORLD
        self.rank = rank
        self.size = size
        self.requests = [MPI.REQUEST_NULL for _ in range(0, size - 1)]

        # Initialize list which stores the corresponding index (0:size-1) to destination relationship
        self.destination = []

        # Store initial solution and objective function, initialize array storing iterate differences and obj. values
        self.x = x
        self.function = function
        self.x_diff = np.empty(0)
        self.obj = [self.function(self.x)]

        # Store SE-ACGD parameters
        self.H = self.function(self.x)
        self.diffH = 0
        self.L = 2
        self.F = ft
        self.tau = tau
        self.T = T

        # Initialize time storing
        self.times = [0.0]
        self.initial_time = 0

        # Initialize buffer for receiving MPI messages
        self.buffer = np.zeros(self.x.shape)

        # Store all the indices each worker will be working on in a list of a list
        self.per_process = int(self.x.shape[0] / (self.size - 1))
        self.worker_idx = []
        for i in range(self.size - 1):
            if i != self.size - 2:
                self.worker_idx.append(range(i * self.per_process, (i + 1) * self.per_process))
            else:
                self.worker_idx.append(range(i * self.per_process, x.shape[0]))

        for i in range(self.per_process):
            self.worker_idx.append((self.rank - 1) * self.per_process + i)

    def send(self, msg, destination):
        """ Send (non-blocking) the next iterate to worker which just finished its work """

        self.comm.isend(msg, dest=destination, tag=self.rank)

    def hamiltonian(self):
        """ Compute the Hamiltonian as defined in the paper """

        return self.function(self.x) + \
               (self.L / (2 * np.sqrt(self.tau))) * np.sum(np.arange(1, self.tau + 1) * self.x_diff[-self.tau:])

    def flush(self):
        for k in range(1000):
            for i in range(0, self.size):
                if i != self.rank:
                    if self.comm.iprobe(source=i, tag=i):
                        self.comm.recv(source=i, tag=i)

    def receive(self, num_updates):
        """ Receive Worker updates for a specified number of updates """

        updates = 0
        while updates < num_updates:
            j = 0
            for i in range(0, self.size):
                if i != self.rank:

                    # Determine which part of the solution the updating worker works on
                    idx = self.worker_idx[j]

                    # Retrieve the message sent from the worker
                    self.buffer[idx] = self.comm.recv(source=i, tag=i)
                    j += 1

            # Update the solution and gradient/objective metrics
            self.x_diff = np.append(self.x_diff, np.linalg.norm(self.buffer))
            self.x += self.buffer
            self.obj.append(self.function(self.x))

            # Send back the new solution to each worker
            for i in range(0, self.size):
                if i != self.rank:
                    self.send(self.x, i)

            # Complete the iteration and time it
            updates += 1
            self.times.append(time.time()-self.initial_time)

        # Compute the new Hamiltonian and difference between Hamiltonians
        newH = self.hamiltonian()
        self.diffH = self.H - newH
        self.H = newH

        # If Hamiltonian decreases sufficiently return a "True" flag, if not then return a "False" flag
        if self.diffH > self.F:
            flag = True
        else:
            flag = False

        # Send the flags to each worker and cancel all older communication requests
        for i in range(0, self.size):
            if i != self.rank:
                self.comm.isend(flag, dest=i, tag=i)

        return flag

    def large_gradient(self):
        """ Large-Gradient sub-algorithm (LG-ACGD) in paper (tau iterations) """

        flag = True
        while flag:
            flag = self.receive(self.tau)
        return

    def perturb(self):
        """ Perturb sub-algorithm (P-ACGD) in paper (T iterations) """

        # Compute perturbation
        unit = np.random.normal(size=self.x.shape)
        unit = unit / np.linalg.norm(unit)
        r = np.random.uniform(low=0, high=1)
        r = r ** (1 / unit.size)
        scale = 0.05
        pert = r * unit * scale

        # Perturb solution
        self.x += pert

        # Compute T ACGD updates
        flag = self.receive(self.T)

        return flag

    def acgd(self):
        """ SE-ACGD Algorithm """

        # While the perturbation algorithm hasn't yet concluded
        flag = True
        self.initial_time = time.time()
        while flag:
            # Run LD-ACGD
            self.large_gradient()
            print('Stationary Point Reached, Need Perturbation')
            print('Current Objective Value: %f' % self.function(self.x))
            print('Current Hamiltonian Value: %f' % self.H)
            print('Current Hamiltonian Difference: %f' % self.diffH)
            print('===================================================')
            # Run P-ACGD
            flag = self.perturb()
            print('Post Perturbation:')
            print('Current Objective Value: %f' % self.function(self.x))
            print('Current Hamiltonian Value: %f' % self.H)
            print('Current Hamiltonian Difference: %f' % self.diffH)
            print('===================================================')
        return self.x, self.x_diff, np.array(self.obj), np.array(self.times)


class ParallelWorker:
    """ Class constructed to turn one process into a parallel worker for SE-ACGD """

    def __init__(self, rank, size, root, x, grad, stepsize, delay):
        """ Initialize Parallel Worker """

        # Initialize MPI variables
        self.comm = MPI.COMM_WORLD
        self.rank = rank
        self.size = size
        self.req = MPI.REQUEST_NULL

        # Initialize message count, root destination (central server), stepsize, gradient function, and initial solution
        self.msg_count = 0
        self.root = root
        self.grad = grad
        self.x = x
        self.stepsize = stepsize

        # User defined worker delay
        self.delay = delay

        # Compute the indices of the solution that the worker will be working on
        per_process = int(self.x.shape[0]/(self.size-1))
        index_list = np.arange(self.size)
        index_list = index_list[index_list != self.root]
        worker_idx = np.where(index_list == self.rank)[0][0]
        if worker_idx == max(index_list):
            self.idx = range(worker_idx * per_process, x.shape[0])
        else:
            self.idx = range(worker_idx * per_process, (worker_idx+1) * per_process)

    def receive(self):
        """ Receive (non-blocking) the next solution from the central server """

        self.req = self.comm.irecv(source=self.root, tag=self.root)

    def send(self, msg):
        """ Send (non-blocking) the computed gradient to the central server """

        self.comm.isend(msg, dest=self.root, tag=self.rank)

    def update(self):
        """ Single Worker Update sub-algorithm (SW-ACGD) """

        return -self.stepsize * self.grad(self.x, self.idx)

    def run(self):
        """ Perform worker updates """

        # Initiate receiving (non-blocking) the end flag (this won't arrive for awhile)
        end_req = self.comm.irecv(source=self.root, tag=self.rank)


        while True:

            # Compute gradient
            msg = self.update()

            # Add in an optional delay (to replicate a slow worker)
            if self.rank == 1:
                time.sleep(self.delay)

            # Send computed gradient
            self.send(msg)

            # Receive next solution (non-blocking)
            self.receive()

            # Wait until the next solution arrives or the end flag arrives
            while True:
                if end_req.Get_status():
                    self.req.cancel()
                    return end_req.wait()
                if self.req.Get_status():
                    self.x = self.req.wait()
                    break

    def acgd(self):
        """ Perform SE-ACGD algorithm """

        # While the perturbation algorithm hasn't yet concluded
        flag2 = True
        while flag2:
            flag1 = True
            while flag1:
                flag1 = self.run()
            flag2 = self.run()
