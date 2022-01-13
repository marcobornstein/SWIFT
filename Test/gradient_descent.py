import numpy as np
import time


def gradient_descent(x0, f, grad, eta, eps, idx, delay, max_iter):
    count = 0
    x = x0
    gradients = [10*eps]
    times = [0.0]
    obj_val = [f(x)]
    initial_time = time.time()
    while gradients[-1] >= eps and count < max_iter:
        time.sleep(delay)
        g = grad(x, idx)
        gradients.append(np.linalg.norm(g))
        x -= eta*g
        count += 1
        obj_val.append(f(x))
        times.append(time.time()-initial_time)
    gradients.pop(0)
    return x, np.array(gradients), np.array(obj_val), np.array(times), count
