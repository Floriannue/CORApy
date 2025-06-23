"""
minkDiff - Minkowski difference

This function computes the Minkowski difference between two sets:
{ s | s ⊕ S₂ ⊆ S₁ }

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-May-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Union, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def minkDiff(S1: 'ContSet', S2: Union['ContSet', np.ndarray], method: Optional[str] = None) -> 'ContSet':
    """
    Computes the Minkowski difference between two sets
    
    Computes the set { s | s ⊕ S₂ ⊆ S₁ }
    
    Args:
        S1: First contSet object
        S2: Second contSet object or numerical vector
        method: Optional method for computation
        
    Returns:
        ContSet: Minkowski difference S₁ ⊖ S₂
        
    Raises:
        CORAerror: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([0.5, 0.5], [1, 1])
        >>> S_diff = minkDiff(S1, S2)
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops',
                   f'minkDiff not implemented for {type(S1).__name__} and {type(S2).__name__}') 