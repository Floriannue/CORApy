"""
empty - instantiates an empty zonotope bundle

Syntax:
    zB = zonoBundle.empty(n)

Inputs:
    n - dimension

Outputs:
    zB - empty zonotope bundle
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck


def empty(n: int = 0):
    """
    Instantiates an empty zonotope bundle
    
    Args:
        n: dimension (default: 0)
        
    Returns:
        zB: empty zonotope bundle
    """
    from .zonoBundle import ZonoBundle
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    # Parse input - match MATLAB behavior
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'nonnegative']]])
    
    # MATLAB: zB = zonoBundle({zonotope(zeros(n,0))});
    # Always create a zonotope (even for n=0) and wrap in list
    Z_empty = Zonotope(np.zeros((n, 0)))
    zB = ZonoBundle([Z_empty])
    
    return zB