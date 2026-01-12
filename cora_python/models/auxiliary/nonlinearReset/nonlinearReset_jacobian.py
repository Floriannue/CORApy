import numpy as np

def nonlinearReset_jacobian(x, u):
    A = np.array([[1, 0],
        [0, 1]])

    B = np.array([[0],
        [0]])

    return A, B
