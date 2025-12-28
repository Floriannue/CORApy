"""
uminus - overloads the unary '-' operator

This function implements the unary minus operation for contSet objects
by multiplying the set by -1.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 06-April-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def uminus(S: 'ContSet') -> 'ContSet':
    """
    Overloads the unary '-' operator
    
    Matches MATLAB exactly: S = -1 * S
    
    Args:
        S: contSet object to negate
        
    Returns:
        ContSet: Negated set (-S)
        
    Example:
        >>> S = zonotope([1, 0], [[1, 0], [0, 1]])
        >>> neg_S = uminus(S)  # or neg_S = -S
    """
    # MATLAB: S = -1 * S
    # This uses the __mul__ operator which calls mtimes for zonotope
    return -1 * S 