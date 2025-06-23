"""
lift_ - lifts a set to a higher-dimensional space (internal use)

This function provides the internal implementation for lifting sets to higher dimensions.
It should be overridden in subclasses to provide specific lifting logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 13-September-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def lift_(S: 'ContSet', N: int, proj: np.ndarray) -> 'ContSet':
    """
    Lifts a set to a higher-dimensional space (internal use)
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific lifting logic.
    
    Args:
        S: contSet object
        N: Dimension of the higher-dimensional space
        proj: States of the high-dimensional space that correspond to the
              states of the low-dimensional set
        
    Returns:
        ContSet: Set in the higher-dimensional space
        
    Raises:
        CORAerror: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S = interval([1, 2], [3, 4])
        >>> S_lifted = lift_(S, 4, np.array([1, 3]))
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops',
                   f'lift_ not implemented for {type(S).__name__} with N={N}') 