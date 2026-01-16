import numpy as np

def out_jacobian_jetEngine(x, u):
    C = np.array([[1.00000000000000, 0],
        [0, 1.00000000000000]])

    D = np.array([[0],
        [0]])

    return C, D
