import numpy as np
from mpi4py import MPI
import time
from communicator import AsyncCentralServer, SyncCentralServer, ParallelWorker
from gradient_descent import gradient_descent
import matplotlib.pyplot as plt


def test_function(x):
    return 0.5 * (x[0] ** 4) - x[0] ** 2 + 0.5 * (x[1] ** 2)


def grad_f(x, idx):
    if len(idx) == 1:
        if idx[0] == 0:
            return 2 * (x[0] ** 3) - 2 * x[0]
        else:
            return x[1]
    else:
        return np.array([2 * (x[0] ** 3) - 2 * x[0], x[1]])


def run_experiment(rank, size, root, initial_solution, delay, grad, stepsize, ft, tau, T, flag):
    if rank == root:

        print('Initial Objective Value: %f' % test_function(initial_solution))
        print('===================================================')

        if flag == 'A':
            server = AsyncCentralServer(rank, size, initial_solution, test_function, ft, tau, T)
        else:
            server = SyncCentralServer(rank, size, initial_solution, test_function, ft, int(tau / 2), int(T / 2))

        t = time.time()
        output, x_diff, objective_value, wall_time = server.acgd()
        total_time = time.time() - t

        server.flush()

        print('Local Minimum Objective Value: %f' % test_function(output))
        if flag == 'A':
            print('Asynchronous Time: %fs' % total_time)
        else:
            print('Synchronous Time: %fs' % total_time)

        print('===================================================')
        print('===================================================')
        print('===================================================')

        return output, x_diff, objective_value, wall_time

    else:

        # Perform ACGD here between other processes
        worker = ParallelWorker(rank, size, root, initial_solution, grad, stepsize, delay)
        worker.acgd()
        return 0, 0, 0, 0


def run(rank, size, stepsize, ft, tau, T):
    root = 0
    grad = grad_f
    delay = 0.000075
    runs = 5
    wt_list_a = []
    obj_list_a = []
    wt_list_b = []
    obj_list_b = []

    MPI.COMM_WORLD.Barrier()

    for _ in range(runs):
        x_init = np.array([0.0, 5.0])
        out_a, xd_a, ov_a, wt_a = run_experiment(rank, size, root, x_init, delay, grad, stepsize, ft, tau, T, 'A')
        obj_list_a.append(ov_a)
        wt_list_a.append(wt_a)
        MPI.COMM_WORLD.Barrier()

    for _ in range(runs):
        x_init = np.array([0.0, 5.0])
        out_b, xd_b, ov_b, wt_b = run_experiment(rank, size, root, x_init, delay, grad, stepsize, ft, tau, T, 'S')
        obj_list_b.append(ov_b)
        wt_list_b.append(wt_b)
        MPI.COMM_WORLD.Barrier()

    if rank == root:

        iterations = xd_a.shape[0]
        idx = [0, 1]
        x_init = np.array([0.0, 5.0])
        t3 = time.time()
        x_gd, grads_gd, obj_gd, times_gd, count_gd = gradient_descent(x_init, test_function, grad, stepsize, ft, idx,
                                                                      delay, iterations)
        gdtime = time.time() - t3

        print('Serial Gradient Descent Objective Value: %f' % test_function(x_gd))
        print('Serial Gradient Descent Time: %fs' % gdtime)
        for k in range(runs):
            plt.plot(wt_list_a[k], obj_list_a[k], color='red',
                     label='Asynchronous Coordinate Gradient Descent (%d Runs)' % runs)
            plt.plot(wt_list_b[k], obj_list_b[k], color='blue',
                     label='Synchronized Coordinate Gradient Descent (%d Runs)' % runs)
        plt.plot(times_gd, obj_gd, color='green', label='Serial Gradient Descent')
        plt.yscale('symlog')
        plt.title('Convergence of Gradient Descent Methods')
        plt.xlabel('Wall Time (Seconds)')
        plt.ylabel('Objective Function Value')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        plt.savefig('Objective-Wall-5Runs-0.000075.png')
        plt.clf()

        '''
        plt.plot(wall_times, obj_vals)
        plt.yscale('symlog')
        plt.title('Convergence of Synchronous CGD')
        plt.xlabel('Wall Time (Seconds)')
        plt.ylabel('Objective Function Value')
        plt.savefig('Synchronous-Objective-Wall-5e5.png')
        plt.clf()

        plt.semilogy(range(1, x_diff.shape[0] + 1), x_diff / stepsize)
        plt.title('Convergence of Synchronous CGD')
        plt.xlabel('Iterations')
        plt.ylabel('Norm of Gradient')
        plt.savefig('Synchronous-Gradient-Norm.png')
        plt.clf()

        plt.plot(range(1, len(obj_vals) + 1), obj_vals)
        plt.title('Convergence of Synchronous CGD')
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function Value')
        plt.savefig('Synchronous-Objective.png')
        plt.clf()

        plt.plot(range(1, count_gd + 1), grads_gd)
        plt.yscale('symlog')
        plt.title('Convergence of Serial Gradient Descent')
        plt.xlabel('Iterations')
        plt.ylabel('Norm of Gradient')
        plt.savefig('Gradient-Descent-Norm.png')
        plt.clf()

        plt.semilogy(range(1, len(obj_gd)+1), obj_gd)
        plt.title('Convergence of Serial Gradient Descent')
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function Value')
        plt.savefig('Gradient-Descent-Objective.png')
        plt.clf()

        plt.semilogy(times_gd, obj_gd)
        plt.title('Convergence of Serial Gradient Descent')
        plt.xlabel('Wall Time (Seconds)')
        plt.ylabel('Objective Function Value')
        plt.savefig('Gradient-Descent-Objective-Wall-5e5.png')
        plt.clf()
        '''


if __name__ == "__main__":

    p_rank = MPI.COMM_WORLD.Get_rank()
    p_size = MPI.COMM_WORLD.Get_size()

    tau = 30
    p_iter = 20000

    old_val = 0
    for i in np.linspace(0, 0.5, 200):
        val = (15 / 8) * tau ** (0.5 - i) - np.sqrt(tau) - 0.5
        if val < 0:
            break
        old_val = val

    beta = old_val

    epsilon = 0.1
    rho = 2
    delta = 0.01
    d = 2
    mu = 2
    del_f = 13
    L = 2
    sigma = max((1280 * np.sqrt(d) * del_f * L * tau) / (delta * np.sqrt(np.pi) * epsilon ** 2), 8)
    iota = mu * np.log2(sigma)
    chi = max(1, np.sqrt(rho * epsilon) / L ** 2)
    eta = 1 / (iota * 2 * L * tau ** (0.5 - beta))
    F = L * (eta ** 2) * (epsilon ** 2) * (1 / (eta * L) - 1 / 2 - np.sqrt(tau))

    run(p_rank, p_size, eta, F, tau, p_iter)
