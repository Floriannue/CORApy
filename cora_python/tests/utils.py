"""
Utility functions for tests

This module provides utility functions commonly used in test cases,
especially functions that replicate MATLAB testing functionality.
"""

import numpy as np


def compareMatrices(A, B, tol=1e-12):
    """
    Compare two matrices/vectors for approximate equality.
    
    This function replicates the MATLAB compareMatrices functionality used in tests.
    
    Args:
        A: First matrix/vector
        B: Second matrix/vector  
        tol: Tolerance for comparison (default: 1e-12)
        
    Returns:
        bool: True if matrices are approximately equal, False otherwise
    """
    if A is None and B is None:
        return True
    if A is None or B is None:
        return False
        
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Check shapes match
    if A.shape != B.shape:
        return False
    
    # Check all elements are close
    return np.allclose(A, B, atol=tol, rtol=tol)


def withinTol(a, b, tol=1e-12):
    """
    Check if two values are within tolerance.
    
    Args:
        a: First value
        b: Second value
        tol: Tolerance (default: 1e-12)
        
    Returns:
        bool: True if values are within tolerance
    """
    return abs(a - b) <= tol


def assertLoop(condition, *args):
    """
    Assert condition with additional debugging info.
    
    This replicates MATLAB's assertLoop functionality used in long tests.
    
    Args:
        condition: Boolean condition to check
        *args: Additional arguments for debugging
    """
    if not condition:
        debug_info = ", ".join(str(arg) for arg in args)
        raise AssertionError(f"Assertion failed with args: {debug_info}")


def isequal(A, B, tol=1e-12):
    """
    Check if two objects are equal (replicating MATLAB isequal).
    
    Args:
        A: First object
        B: Second object
        tol: Tolerance for numeric comparison
        
    Returns:
        bool: True if objects are equal
    """
    return compareMatrices(A, B, tol) 