import numpy as np
from cora_python.contSet.interval import Interval

def nonlinearReset_hessian(x, u):
    Hx = Interval(np.array([[0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]))

    Hu = Interval(np.array([[0],
        [0],
        [0],
        [0],
        [0],
        [0]]))

    return Hx, Hu
