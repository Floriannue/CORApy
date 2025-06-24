"""
eq - overloads '==' operator to check if two linear systems are equal

This function checks if two linearSys objects are equal by comparing
all their matrices and properties within a specified tolerance.

Authors: Florian Nüssel (Python implementation)
Date: 2025-06-08
"""

from typing import TYPE_CHECKING

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
    return linsys1.isequal(linsys2, tol) 