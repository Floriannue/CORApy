"""
uplus - overloads the unary '+' operator

This function implements the unary plus operation for contSet objects.
The unary plus operation returns the set unchanged.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 06-April-2023 (MATLAB)
Python translation: 2025
"""


def uplus(S: 'ContSet') -> 'ContSet':
    """
    Overloads the unary '+' operator
    
    Args:
        S: contSet object
        
    Returns:
        ContSet: The same set (S = +S)
        
    Example:
        >>> S = zonotope([1, 0], [[1, 0], [0, 1]])
        >>> pos_S = uplus(S)  # or pos_S = +S
        >>> # pos_S is the same as S
    """
    # Unary plus returns the set unchanged
    return S