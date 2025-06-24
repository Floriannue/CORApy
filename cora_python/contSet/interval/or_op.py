"""
or_op - computes the union of an interval and a set or point

Syntax:
    S_out = I | S
    S_out = or_op(I, S)
    S_out = or_op(I, S, mode)

Inputs:
    I - interval object
    S - contSet object
    mode - 'exact', 'outer', 'inner'

Outputs:
    S_out - union of the two sets

Example: 
    I1 = interval([-2, -2], [-1, -1])
    I2 = interval([0, 0], [2, 2])
    res = I1 | I2

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np
from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def or_op(I, S, mode='outer'):
    """
    Computes the union of an interval and a set or point
    
    Args:
        I: interval object
        S: contSet object or numeric
        mode: 'exact', 'outer', 'inner'
        
    Returns:
        S_out: union of the two sets
    """
    
    # Check dimensions
    if hasattr(S, 'dim') and I.dim() != S.dim():
        raise CORAerror("CORA:dimensionMismatch", "Dimensions must match for union operation")
    
    # Call function with lower precedence
    if hasattr(S, 'precedence') and hasattr(I, 'precedence') and S.precedence < I.precedence:
        return S.or_op(I, mode)
    
    # Empty set case: union is the other set
    if hasattr(S, 'representsa_') and S.representsa_('emptySet', 1e-14):
        return I
    elif I.representsa_('emptySet', 1e-14):
        return S
    
    # Interval-interval case
    if hasattr(S, '__class__') and S.__class__.__name__ == 'Interval':
        return Interval(np.minimum(I.inf, S.inf), np.maximum(I.sup, S.sup))
    
    # Numeric case
    if isinstance(S, (int, float, np.ndarray)):
        S = np.atleast_1d(S)
        return Interval(np.minimum(I.inf, S), np.maximum(I.sup, S))
    
    raise CORAerror('CORA:noops', f"Union operation not supported between {type(I)} and {type(S)}") 