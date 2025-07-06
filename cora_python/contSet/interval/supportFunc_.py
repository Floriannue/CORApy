import numpy as np
from .interval import Interval

def supportFunc_(I, dir, type, *varargin):
    """
    Calculate the upper or lower bound of an interval along a
    certain direction
    """

    # special handling for empty set
    if I.is_empty():
        x = []
        if type == 'upper':
            val = -np.inf
        elif type == 'lower':
            val = np.inf
        elif type == 'range':
            val = Interval(-np.inf, np.inf)
        return val, x

    # take infimum/supremum depending on sign of direction; for entries with 0,
    # it does not matter
    idx = np.sign(dir) == -1
    if type == 'upper':
        x = I.sup.copy()
        x[idx] = I.inf[idx]
        val = dir.T @ x
    elif type == 'lower':
        x = I.inf.copy()
        x[idx] = I.sup[idx]
        val = dir.T @ x
    elif type == 'range':
        x = np.array([I.inf, I.sup]).T
        x[idx, 0] = I.sup[idx]
        x[idx, 1] = I.inf[idx]
        val = Interval(dir.T @ x[:, 0], dir.T @ x[:, 1])
        
    return val, x 