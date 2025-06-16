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
    
    This function delegates to the object's uminus method if available,
    otherwise implements as multiplication by -1.
    
    Args:
        S: contSet object to negate
        
    Returns:
        ContSet: Negated set (-S)
        
    Example:
        >>> S = zonotope([1, 0], [[1, 0], [0, 1]])
        >>> neg_S = uminus(S)  # or neg_S = -S
    """
    # Check if the object has an uminus method and use it
    if hasattr(S, 'uminus') and callable(getattr(S, 'uminus')):
        return S.uminus()
    
    # Fallback: implement as multiplication by -1
    from .times import times
    return times(-1, S) 