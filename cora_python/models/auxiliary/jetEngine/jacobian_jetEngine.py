import numpy as np

def jacobian_jetEngine(x, u):
    A = np.array([[-1.5*x[0, 0]**2 - 3.0*x[0, 0], -1],
        [3, -1]])

    B = np.array([[0],
        [0]])

    return A, B
