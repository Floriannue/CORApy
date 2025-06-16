"""
times - overloaded '.*' operator for element-wise multiplication with zonotope

Syntax:
    Z = times(factor1, factor2)
    Z = factor1 .* factor2

Inputs:
    factor1 - zonotope object or numeric scalar
    factor2 - zonotope object or numeric scalar

Outputs:
    Z - resulting zonotope object

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 22-March-2007 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union


def times(factor1: Union['Zonotope', float, int], factor2: Union['Zonotope', float, int]) -> 'Zonotope':
    """
    Element-wise multiplication for zonotope objects
    
    Args:
        factor1: zonotope object or numeric scalar
        factor2: zonotope object or numeric scalar
        
    Returns:
        Zonotope: Resulting zonotope object
    """
    from .zonotope import Zonotope
    
    # scalar * zonotope
    if isinstance(factor1, (int, float, np.number)) and isinstance(factor2, Zonotope):
        scalar = factor1
        
        # Handle empty zonotope case
        if factor2.is_empty():
            return Zonotope.empty(factor2.dim())
        
        # Scale center and generators
        c = scalar * factor2.c
        G = scalar * factor2.G if factor2.G.size > 0 else factor2.G
        
        return Zonotope(c, G)
    
    # zonotope * scalar
    elif isinstance(factor1, Zonotope) and isinstance(factor2, (int, float, np.number)):
        scalar = factor2
        
        # Handle empty zonotope case
        if factor1.is_empty():
            return Zonotope.empty(factor1.dim())
        
        # Scale center and generators
        c = scalar * factor1.c
        G = scalar * factor1.G if factor1.G.size > 0 else factor1.G
        
        return Zonotope(c, G)
    
    else:
        raise TypeError(f"Unsupported operand types for .*: {type(factor1)} and {type(factor2)}") 