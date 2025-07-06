import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def enlarge(I, factor):
    """
    Enlarges an interval object around its center.
    """
    # get center and radius
    c = I.center()
    r = I.rad()

    # handle unbounded dimensions
    is_unbounded = np.isinf(r)
    if np.any(is_unbounded & (factor < 1)):
        # if factor < 1, there is no center -> return error
        raise CORAerror('CORA:notSupported', 'Factor must be finite and non-zero for unbounded intervals.')
    
    # Ensure factor is a numpy array
    factor = np.asarray(factor)
    
    # Ensure factor has the same shape as r to avoid broadcasting issues
    if factor.ndim == 1 and r.ndim == 2:
        factor = factor.reshape(-1, 1)
    
    # enlarged intervals
    inf = c - r * factor
    sup = c + r * factor

    # factor > 1, dimension expands to [-Inf,Inf]
    expands_to_inf = is_unbounded & (factor > 1)
    inf[expands_to_inf] = -np.inf
    sup[expands_to_inf] = np.inf

    return Interval(inf, sup) 