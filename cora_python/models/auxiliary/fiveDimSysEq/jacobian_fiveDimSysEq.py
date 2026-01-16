import numpy as np

def jacobian_fiveDimSysEq(x, u):
    A = np.array([[-1, -4, 0, 0, 0],
        [4, -1, 0, 0, 0],
        [0, 0, -3, 1, 0],
        [0, 0, -1, -3, 0],
        [0, 0, 0, 0, -2]])

    B = np.array([[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]])

    return A, B
