import numpy as np

def jacobian_jetEngine(x, u):
    A = np.array([[0, 1],
        [-1, -1]])

    B = np.array([[0],
        [1]])

    return A, B
