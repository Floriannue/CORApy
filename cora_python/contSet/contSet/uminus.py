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
    # Check if subclass has overridden uminus method
    base_class = type(S).__bases__[0] if type(S).__bases__ else None
    if (hasattr(type(S), 'uminus') and 
        base_class and hasattr(base_class, 'uminus') and
        type(S).uminus is not base_class.uminus):
        return type(S).uminus(S)
    
    # Fallback: implement as multiplication by -1
    return S.times(-1) 