"""
ne - overloads '!=' operator to check if two linear systems are not equal

This function checks if two linearSys objects are not equal by negating
the result of the equality comparison.

Authors: Florian Nüssel (Python implementation)
Date: 2025-06-08
"""

from typing import TYPE_CHECKING
from .eq import eq

if TYPE_CHECKING:
    from .linearSys import LinearSys


def ne(linsys1: 'LinearSys', linsys2: 'LinearSys', tol: float = None) -> bool:
    """
    Overloads '!=' operator to check if two linear systems are not equal
    
    Args:
        linsys1: First linearSys object
        linsys2: Second linearSys object
        tol: Tolerance for numerical comparison (default: machine epsilon)
        
    Returns:
        bool: True if systems are not equal, False otherwise
        
    Example:
        linsys1 = LinearSys([[1]], [[0]])
        linsys2 = LinearSys([[1]], [[1]])
        res = ne(linsys1, linsys2)  # True
    """
    return not eq(linsys1, linsys2, tol) 