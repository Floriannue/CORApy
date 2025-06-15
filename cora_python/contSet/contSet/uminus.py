"""
uminus - overloads the unary '-' operator

This function implements the unary minus operation for contSet objects
by multiplying the set by -1.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 06-April-2023 (MATLAB)
Python translation: 2025
"""


def uminus(S: 'ContSet') -> 'ContSet':
    """
    Overloads the unary '-' operator
    
    Args:
        S: contSet object to negate
        
    Returns:
        ContSet: Negated set (-S)
        
    Example:
        >>> S = zonotope([1, 0], [[1, 0], [0, 1]])
        >>> neg_S = uminus(S)  # or neg_S = -S
    """
    # Implement as multiplication by -1
    from .times import times
    return times(-1, S) 