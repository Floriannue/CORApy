"""
isequal - checks if two zonotope objects are equal

Syntax:
    res = isequal(Z1, Z2)
    res = isequal(Z1, Z2, tol)

Inputs:
    Z1 - zonotope object
    Z2 - zonotope object
    tol - tolerance (optional, default: 1e-12)

Outputs:
    res - true/false

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 22-March-2007 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Any, Optional


def isequal(Z1: 'Zonotope', Z2: Any, tol: Optional[float] = None) -> bool:
    """
    Checks if two zonotope objects are equal
    
    Args:
        Z1: First zonotope object
        Z2: Second object to compare with
        tol: Tolerance for comparison (default: 1e-12)
        
    Returns:
        bool: True if zonotopes are equal, False otherwise
    """
    from .zonotope import Zonotope
    
    # Set default tolerance
    if tol is None:
        tol = 1e-12
    
    # Check if Z2 is also a zonotope
    if not isinstance(Z2, Zonotope):
        return False
    
    # Check if both are empty
    if Z1.is_empty() and Z2.is_empty():
        return Z1.dim() == Z2.dim()
    
    # Check if one is empty and the other is not
    if Z1.is_empty() or Z2.is_empty():
        return False
    
    # Check dimensions
    if Z1.dim() != Z2.dim():
        return False
    
    # Check centers
    if not np.allclose(Z1.c, Z2.c, atol=tol, rtol=tol):
        return False
    
    # Check generator matrices
    if Z1.G.shape != Z2.G.shape:
        return False
    
    if not np.allclose(Z1.G, Z2.G, atol=tol, rtol=tol):
        return False
    
    return True 