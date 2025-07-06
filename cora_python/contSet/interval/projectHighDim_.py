import numpy as np
from cora_python.contSet.interval.interval import Interval

def projectHighDim_(I, N, proj):
    """
    Projects an interval set to a higher-dimensional space.
    """
    if I.is_empty():
        return Interval.empty(N)

    # init bounds
    lb = np.zeros((N, 1))
    ub = np.zeros((N, 1))

    # project
    # The public method already converted proj to 0-based indices
    lb[proj] = I.inf
    ub[proj] = I.sup

    # result
    return Interval(lb, ub) 