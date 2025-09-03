"""
interval - Over-approximates a polynomial zonotope by an interval

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Niklas Kochdumper
         Python: AI Assistant
"""

import numpy as np
from typing import Union, Optional
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.helper.sets.contSet.polyZonotope.poly2bernstein import poly2bernstein
from .supportFunc_ import supportFunc_


def interval(pZ, method: str = 'interval', splits: int = 8) -> Interval:
    """
    Over-approximates a polynomial zonotope by an interval.
    
    Args:
        pZ: polyZonotope object
        method: method used to calculate the bounds for all dimensions
                'interval': interval arithmetic
                'split': split set multiple times
                'bnb': taylor models with "branch and bound" algorithm
                'bnbAdv': taylor models with advanced bnb-algorithm
                'globOpt': verified global optimization 
                'bernstein': conversion to a bernstein polynomial
        splits: number of splits for 'split'
        
    Returns:
        I: interval object
        
    See also: zonotope, supportFunc
    """
    
    # Check input arguments
    if not hasattr(pZ, 'c') or not hasattr(pZ, 'G') or not hasattr(pZ, 'GI') or not hasattr(pZ, 'E'):
        raise ValueError("Input must be a polyZonotope object")
    
    valid_methods = ['interval', 'split', 'bnb', 'bnbAdv', 'globOpt', 'bernstein']
    if method not in valid_methods:
        raise ValueError(f"Invalid method. Must be one of: {valid_methods}")
    
    # Compute over-approximating interval with the selected method
    if method == 'interval':
        # Convert to zonotope and then to interval
        z = Zonotope(pZ.c, np.hstack([pZ.G, pZ.GI]))
        return z.interval()
        
    elif method == 'bernstein':
        # Use Bernstein polynomial conversion
        p = pZ.E.shape[0]
        dom = Interval(-np.ones(p), np.ones(p))
        
        # Dependent generators: convert to bernstein polynomial
        B = poly2bernstein(pZ.G, pZ.E, dom)
        
        # Compute bounds from Bernstein coefficients
        if len(B) > 0 and B[0].size > 0:
            infi = [np.min(B[i]) for i in range(len(B))]
            sup = [np.max(B[i]) for i in range(len(B))]
            I1 = Interval(infi, sup)
        else:
            I1 = Interval(np.zeros(pZ.c.shape[0]), np.zeros(pZ.c.shape[0]))
        
        # Independent generators: enclose zonotope with interval
        if pZ.GI.size > 0:
            I2 = Zonotope(pZ.c, pZ.GI).interval()
        else:
            I2 = Interval(pZ.c.flatten(), pZ.c.flatten())
        
        return I1 + I2
        
    else:
        # Use supportFunc_ for other methods (split, bnb, bnbAdv, globOpt)
        n = len(pZ.c)
        e = np.zeros(n)
        I = Interval(e, e)
        
        # Loop over all system dimensions
        for i in range(n):
            e_ = np.zeros(n)
            e_[i] = 1
            I[i] = supportFunc_(pZ, e_, 'range', method, splits, 1e-3)
        
        return I
