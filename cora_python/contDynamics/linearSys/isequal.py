"""
isequal - checks if two linear systems are equal

This function checks if two linearSys objects are equal by comparing
all their matrices and properties within a specified tolerance.

Authors: Florian Nüssel (Python implementation)
Date: 2025-06-08
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.check import withinTol
from cora_python.g.functions.matlab.validate.check import compareMatrices

if TYPE_CHECKING:
    from .linearSys import LinearSys


def isequal(linsys1: 'LinearSys', linsys2: 'LinearSys', tol: float = None) -> bool:
    """
    Checks if two linear systems are equal
    
    Args:
        linsys1: First linearSys object
        linsys2: Second linearSys object
        tol: Tolerance for numerical comparison (default: machine epsilon)
        
    Returns:
        bool: True if systems are equal, False otherwise
        
    Example:
        A = [[2, 1], [-1, 2]]
        B = [[1], [-1]]
        linsys1 = LinearSys(A, B)
        linsys2 = LinearSys(A, np.array(B) + 1e-14)
        res = isequal(linsys1, linsys2)  # True (within tolerance)
    """
    
    # Set default tolerance
    if tol is None:
        tol = np.finfo(float).eps
    
    # Check if both are LinearSys objects
    if not hasattr(linsys1, 'A') or not hasattr(linsys2, 'A'):
        return False
    
    # Check name
    if linsys1.name != linsys2.name:
        return False
    
    # Check dimensions
    if (linsys1.nr_of_dims != linsys2.nr_of_dims or 
        linsys1.nr_of_inputs != linsys2.nr_of_inputs or
        linsys1.nr_of_outputs != linsys2.nr_of_outputs):
        return False
    
    # Check system matrix A
    if not compareMatrices(linsys1.A, linsys2.A, tol, ordered=True, signed=True):
        return False
    
    # Check input matrix B
    if not compareMatrices(linsys1.B, linsys2.B, tol, ordered=True, signed=True):
        return False
    
    # Check offset c (differential equation)
    result_c = withinTol(linsys1.c, linsys2.c, tol)
    if not (result_c if isinstance(result_c, bool) else result_c.all()):
        return False
    
    # Check output matrix C
    result_C = withinTol(linsys1.C, linsys2.C, tol)
    if not (result_C if isinstance(result_C, bool) else result_C.all()):
        return False
    
    # Check feedthrough matrix D
    result_D = withinTol(linsys1.D, linsys2.D, tol)
    if not (result_D if isinstance(result_D, bool) else result_D.all()):
        return False
    
    # Check offset k (output equation)
    result_k = withinTol(linsys1.k, linsys2.k, tol)
    if not (result_k if isinstance(result_k, bool) else result_k.all()):
        return False
    
    # Check disturbance matrix E (state)
    if not compareMatrices(linsys1.E, linsys2.E, tol, ordered=True, signed=True):
        return False
    
    # Check disturbance matrix F (output)
    if not compareMatrices(linsys1.F, linsys2.F, tol, ordered=True, signed=True):
        return False
    
    # All checks passed
    return True 