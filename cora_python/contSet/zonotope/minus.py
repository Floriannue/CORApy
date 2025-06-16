"""
minus - overloaded '-' operator for zonotope objects

Syntax:
    Z = Z1 - Z2
    Z = Z1 - p

Inputs:
    Z1 - zonotope object
    Z2 - zonotope object or numeric vector

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


def minus(Z1: 'Zonotope', Z2: Union['Zonotope', np.ndarray]) -> 'Zonotope':
    """
    Overloaded '-' operator for zonotope objects
    
    Args:
        Z1: First zonotope object
        Z2: Second zonotope object or numeric vector
        
    Returns:
        Zonotope: Resulting zonotope object
    """
    from .zonotope import Zonotope
    
    if isinstance(Z2, (np.ndarray, list, tuple)) or np.isscalar(Z2):
        # Subtraction with vector: Z1 - p = Z1 + (-p)
        from .plus import plus
        return plus(Z1, -np.array(Z2))
    
    elif hasattr(Z2, 'c') and hasattr(Z2, 'G'):
        # Zonotope-to-zonotope subtraction: Z1 - Z2 = Z1 + (-Z2)
        # This implements the Minkowski difference approximation
        from .plus import plus
        
        # Create -Z2 by negating center and generators
        neg_Z2_c = -Z2.c
        neg_Z2_G = -Z2.G if Z2.G.size > 0 else np.zeros((Z2.c.shape[0], 0))
        
        neg_Z2 = Zonotope(neg_Z2_c, neg_Z2_G)
        
        # Compute Z1 + (-Z2)
        return plus(Z1, neg_Z2)
    
    else:
        raise TypeError(f"Unsupported operand type for -: zonotope and {type(Z2)}") 