import numpy as np

def jacobian_system_dynamics(x, u):
    A = np.array([[0, -x[2, 0], -x[1, 0]],
        [-1, 0, 0],
        [-x[1, 0], -x[0, 0], 0]])

    B = np.array([[0],
        [1],
        [0]])

    return A, B
