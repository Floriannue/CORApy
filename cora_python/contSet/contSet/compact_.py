"""
compact_ - removes redundancies in the representation of a set (internal use)

This function provides the internal implementation for set compactification.
It should be overridden in subclasses to provide specific compactification logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Optional
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def compact_(S: 'ContSet', method: Optional[str] = None, tol: Optional[float] = None) -> 'ContSet':
    """
    Removes redundancies in the representation of a set (internal use)
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific compactification logic.
    
    Args:
        S: contSet object
        method: Method for redundancy removal
        tol: Tolerance for redundancy removal
        
    Returns:
        ContSet: Compacted set
        
    Raises:
        CORAerror: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S = zonotope(center, generators)
        >>> S_compact = compact_(S, 'zeros', 1e-10)
    """
    base_class = type(S).__bases__[0] if type(S).__bases__ else None
    if (hasattr(type(S), 'compact_') and 
        base_class and hasattr(base_class, 'compact_') and
        type(S).compact_ is not base_class.compact_):
        return type(S).compact_(method, tol)
    else:
        # This is overridden in subclass if implemented; throw error
        raise CORAerror('CORA:noops',
                        f'compact_ not implemented for {type(S).__name__} with method {method}') 