import numpy as np
from cora_python.contSet.interval.interval import Interval

def dirac_op(self, *varargin):
    """
    Overloaded built-in dirac function.
    Note: Higher-order derivatives of the dirac function are not implemented
    in the same way as MATLAB's symbolic dirac.

    Element-wise computation:
       dirac(I)   = [0,0]       if 0 not in I
                    [0,Inf]     otherwise
       dirac(1,I) = [0,0]       if 0 not in I
                    [-Inf,Inf]  if 0 in I
                    [-Inf,0]    if 0 = min(I)
                    [0,Inf]     if 0 = max(I)
    """
    nargin = len(varargin)
    I = self

    if nargin == 0:
        n = 0
    elif nargin == 1:
        n = varargin[0]
    else:
        raise ValueError("Too many input arguments.")

    # init with zeros
    res = Interval(np.zeros_like(I.inf), np.zeros_like(I.sup))

    # sign function to check which dimensions contain 0
    signI = I.sign()

    # set the upper bound of those dimensions to Inf
    contains_zero = np.abs(signI.inf + signI.sup) <= 1
    
    if n == 0:
        res.sup[contains_zero] = np.inf
    elif n == 1:
        # [-Inf,Inf] if 0 in I (interior)
        interior_zero = signI.inf + signI.sup == 0
        res.inf[interior_zero] = -np.inf
        res.sup[interior_zero] = np.inf
        
        # [-Inf,0] if 0 = min(I)
        min_zero = signI.inf + signI.sup == 1
        res.inf[min_zero] = -np.inf
        
        # [0,Inf] if 0 = max(I)
        max_zero = signI.inf + signI.sup == -1
        res.sup[max_zero] = np.inf
        
    else: # n > 1
        res.inf[contains_zero] = -np.inf
        res.sup[contains_zero] = np.inf
        
    return res 