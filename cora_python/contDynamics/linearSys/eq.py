"""
eq - overloads '==' operator to check if two linear systems are equal

This function checks if two linearSys objects are equal by comparing
all their matrices and properties within a specified tolerance.

Authors: Florian Nüssel (Python implementation)
Date: 2025-06-08
"""

import numpy as np
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .linearSys import LinearSys


def eq(linsys1: 'LinearSys', linsys2: 'LinearSys', tol: float = None) -> bool:
    """
    Overloads '==' operator to check if two linear systems are equal
    
    Args:
        linsys1: First linearSys object
        linsys2: Second linearSys object  
        tol: Tolerance for numerical comparison (default: machine epsilon)
        
    Returns:
        bool: True if systems are equal, False otherwise
        
    Example:
        linsys1 = LinearSys([[1]], [[0]])
        linsys2 = LinearSys([[1]], [[1]])
        res = eq(linsys1, linsys2)  # False
    """
    return isequal(linsys1, linsys2, tol)


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
    if not _compare_matrices(linsys1.A, linsys2.A, tol):
        return False
    
    # Check input matrix B
    if not _compare_matrices(linsys1.B, linsys2.B, tol):
        return False
    
    # Check offset c (differential equation)
    if not _within_tol(linsys1.c, linsys2.c, tol):
        return False
    
    # Check output matrix C
    if not _within_tol(linsys1.C, linsys2.C, tol):
        return False
    
    # Check feedthrough matrix D
    if not _within_tol(linsys1.D, linsys2.D, tol):
        return False
    
    # Check offset k (output equation)
    if not _within_tol(linsys1.k, linsys2.k, tol):
        return False
    
    # Check disturbance matrix E (state)
    if not _compare_matrices(linsys1.E, linsys2.E, tol):
        return False
    
    # Check disturbance matrix F (output)
    if not _compare_matrices(linsys1.F, linsys2.F, tol):
        return False
    
    # All checks passed
    return True


def _compare_matrices(mat1: np.ndarray, mat2: np.ndarray, tol: float) -> bool:
    """
    Compare two matrices within tolerance
    
    Args:
        mat1: First matrix
        mat2: Second matrix
        tol: Tolerance for comparison
        
    Returns:
        bool: True if matrices are equal within tolerance
    """
    # Handle empty matrices
    if mat1.size == 0 and mat2.size == 0:
        return True
    if mat1.size == 0 or mat2.size == 0:
        return False
    
    # Check shapes
    if mat1.shape != mat2.shape:
        return False
    
    # Check values within tolerance
    return np.allclose(mat1, mat2, atol=tol, rtol=tol)


def _within_tol(arr1: np.ndarray, arr2: np.ndarray, tol: float) -> bool:
    """
    Check if all elements of two arrays are within tolerance
    
    Args:
        arr1: First array
        arr2: Second array
        tol: Tolerance for comparison
        
    Returns:
        bool: True if all elements are within tolerance
    """
    # Handle empty arrays
    if arr1.size == 0 and arr2.size == 0:
        return True
    if arr1.size == 0 or arr2.size == 0:
        return False
    
    # Check shapes
    if arr1.shape != arr2.shape:
        return False
    
    # Check all elements within tolerance
    return np.all(np.abs(arr1 - arr2) <= tol) 