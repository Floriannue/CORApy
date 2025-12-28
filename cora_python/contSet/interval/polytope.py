import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def polytope(I):
    """
    Convert an interval to a polytope.
    """
    from cora_python.contSet.polytope.polytope import Polytope
    # Check if not a matrix set
    if len(I.inf.shape) > 1 and I.inf.shape[1] > 1:
        raise CORAerror('CORA:wrongValue', 'first', 'Interval must not be an n-d array with n > 1.')

    # Get dimension
    n = I.dim()
    
    # Construct halfspace constraints C*x <= d
    C = np.vstack((np.eye(n), -np.eye(n)))
    d = np.vstack((I.sup, -I.inf))
    
    # Eliminate unbounded directions
    idx_inf = np.isinf(d).flatten()
    if np.any(idx_inf):
        C = C[~idx_inf, :]
        d = d[~idx_inf]
    # Ensure d is column vector (flatten first if needed)
    d = d.flatten().reshape(-1, 1)
    
    # Construct polytope object
    return Polytope(C, d) 