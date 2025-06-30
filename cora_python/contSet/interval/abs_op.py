import numpy as np
from cora_python.contSet.interval.interval import Interval

def abs_op(I):
    # abs - returns the absolute value of an interval
    #
    # Syntax:
    #    res = abs(I)
    #
    # Inputs:
    #    I - interval object
    #
    # Outputs:
    #    res - interval object
    #
    # Example:
    #    I = interval(np.array([[-2],[-1]]),np.array([[3],[4]]))
    #    res = abs(I)
    
    # Authors:       Matthias Althoff, Mark Wetzlinger
    # Written:       26-June-2015
    # Last update:   14-February-2015
    #                12-October-2015
    #                27-May-2024 (MW, simplify and speed up code)
    # Last revision: ---
    
    # ------------------------------ BEGIN CODE -------------------------------
    
    # The new lower bound
    # new_inf is 0 if interval contains 0
    # new_inf is I.inf if interval is positive
    # new_inf is -I.sup if interval is negative
    # The MATLAB implementation uses a clever trick to compute this in one line:
    new_inf = np.maximum.reduce([np.zeros_like(I.inf, dtype=float), I.inf, -I.sup])
    
    # The new upper bound is max(|inf|,|sup|)
    new_sup = np.maximum(np.abs(I.inf), np.abs(I.sup))

    return Interval(new_inf, new_sup)
    
# ------------------------------ END OF CODE ------------------------------ 