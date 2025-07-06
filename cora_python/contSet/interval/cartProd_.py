import numpy as np
from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def cartProd_(I, S, *varargin):
    # This is the internal implementation for an Interval instance, so 'I' is self.
    # The public-facing cartProd has already dispatched to this method on an Interval object.
    
    # S can be another Interval or a numeric type
    if isinstance(S, Interval):
        if I.is_empty():
            return S
        elif S.is_empty():
            return I
        # vertcat
        if I.inf.shape[1] == 1 and S.inf.shape[1] == 1:
            return Interval(np.vstack([I.inf, S.inf]), np.vstack([I.sup, S.sup]))
        # horzcat
        elif I.inf.shape[0] == 1 and S.inf.shape[0] == 1:
            return Interval(np.hstack([I.inf, S.inf]), np.hstack([I.sup, S.sup]))
        else:
            raise CORAerror('CORA:dimensionMismatch', I, S)

    elif isinstance(S, (np.ndarray, list)):
        S_arr = np.array(S)
        if S_arr.ndim == 1:
            S_arr = S_arr.reshape(-1, 1)
        return Interval(np.vstack([I.inf, S_arr]), np.vstack([I.sup, S_arr]))

    # Handle the case where the first argument was numeric and passed to S
    elif isinstance(I, (np.ndarray, list)):
        I_arr = np.array(I)
        if I_arr.ndim == 1:
           I_arr = I_arr.reshape(-1, 1)
        return Interval(np.vstack([I_arr, S.inf]), np.vstack([I_arr, S.sup]))

    else:
        # throw error for given arguments
        raise CORAerror('CORA:noops', I, S) 