import numpy as np
from .interval import Interval

def subsref(I, S):
    """
    subsref - Overloads the operator that selects elements, e.g., I(1,2),
    where the element of the first row and second column is referred to.

    Syntax:
        newObj = subsref(I,S)

    Inputs:
        I - interval object
        S - contains information of the type and content of element selections

    Outputs:
        newObj - element or elemets of the interval matrix
    """
    
    # obtain sub-intervals from the interval object
    # MATLAB uses 2D indexing even for scalar/column vectors; emulate that here
    if isinstance(S, tuple) and len(S) == 2 and np.ndim(I.inf) == 1:
        # Treat 1D interval data as a column vector for 2D indexing
        inf_view = np.asarray(I.inf).reshape(-1, 1)
        sup_view = np.asarray(I.sup).reshape(-1, 1)
        new_inf = inf_view[S]
        new_sup = sup_view[S]
    else:
        new_inf = I.inf[S]
        new_sup = I.sup[S]
    
    return Interval(new_inf, new_sup) 