"""
uminus - overloaded unary '-' operator for zonotope objects

Syntax:
    Z = -Z1
    Z = uminus(Z1)

Inputs:
    Z1 - zonotope object

Outputs:
    Z - resulting zonotope object

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 22-March-2007 (MATLAB)
Python translation: 2025
"""

import numpy as np


def uminus(Z: 'Zonotope') -> 'Zonotope':
    """
    Overloaded unary '-' operator for zonotope objects
    
    Args:
        Z: zonotope object to negate
        
    Returns:
        Zonotope: Negated zonotope object (-Z)
    """
    from .zonotope import Zonotope
    
    # Handle empty zonotope case
    if Z.is_empty():
        return Zonotope.empty(Z.dim())
    
    # Negate center and generators
    c = -Z.c
    G = -Z.G if Z.G.size > 0 else Z.G
    
    return Zonotope(c, G) 