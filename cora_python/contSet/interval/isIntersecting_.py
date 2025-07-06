import numpy as np
from cora_python.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.interval.interval import Interval

def isIntersecting_(I, S, type='exact', tol=1e-9, *varargin):
    from cora_python.contSet.contSet import ContSet

    # ensure that numeric is second input argument
    if isinstance(I, np.ndarray) and isinstance(S, ContSet):
        temp = I
        I = S
        S = temp
    
    # numeric case: check containment first
    if isinstance(S, np.ndarray):
        # contains_ returns a tuple (res, cert, scaling), we only need the result
        res, _, _ = I.contains_(S)
        return res

    # call function with lower precedence
    if isinstance(S, ContSet) and S.precedence < I.precedence:
        return S.isIntersecting_(I, type, tol)

    # sets must not be empty
    if I.is_empty() or S.is_empty():
        return False
    
    # interval and interval intersection
    if isinstance(S, Interval):
        
        # Check for intersection in all dimensions
        inf1, sup1 = I.inf, I.sup
        inf2, sup2 = S.inf, S.sup
        
        # The intervals intersect if and only if for all dimensions i,
        # (inf1[i] <= sup2[i]) and (inf2[i] <= sup1[i]), including tolerance.
        intersect = np.logical_and(
            np.logical_or(inf1 <= sup2, np.isclose(inf1, sup2, atol=tol)),
            np.logical_or(inf2 <= sup1, np.isclose(inf2, sup1, atol=tol))
        )
        
        return np.all(intersect)

    raise CORAerror('CORA:noops', I, S) 