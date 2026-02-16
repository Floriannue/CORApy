import numpy as np

def jacobian_vanderPolEq(x, u):
    A = np.array([[0, 1],
        [-2*x[0, 0]*x[1, 0] - 1, 1 - x[0, 0]**2]])

    B = np.array([[0],
        [1]])

    return A, B
