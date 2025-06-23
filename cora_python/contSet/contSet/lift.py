"""
lift - lifts a set to a higher-dimensional space, having the new dimensions unbounded

This function lifts a set from a lower-dimensional space to a higher-dimensional
space by specifying which dimensions in the higher space correspond to the
original set dimensions.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 13-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def lift(S: 'ContSet', N: int, proj: Optional[np.ndarray] = None) -> 'ContSet':
    """
    Lifts a set to a higher-dimensional space, having the new dimensions unbounded
    
    Args:
        S: contSet object
        N: Dimension of the higher-dimensional space
        proj: States of the high-dimensional space that correspond to the
              states of the low-dimensional set (default: 1:dim(S))
        
    Returns:
        ContSet: Set in the higher-dimensional space
        
    Raises:
        CORAerror: If dimensions are invalid
        ValueError: If invalid parameters
        
    Example:
        >>> S = interval([1, 2], [3, 4])  # 2D set
        >>> S_lifted = lift(S, 4, np.array([1, 3]))  # Lift to 4D space at dimensions 1 and 3
    """
    # Set default projection
    if proj is None:
        proj = np.arange(1, S.dim() + 1)  # 1-indexed like MATLAB
    
    # Validate inputs
    if not isinstance(N, int) or N < 0:
        raise ValueError("N must be a non-negative integer")
    
    proj = np.asarray(proj)
    if proj.ndim != 1:
        raise ValueError("proj must be a vector")
    
    # Check dimension constraints
    if S.dim() > N:
        raise CORAerror('CORA:wrongValue',
                       'Dimension of higher-dimensional space must be larger than or equal to the dimension of the given set.')
    
    if S.dim() != len(proj):
        raise CORAerror('CORA:wrongValue',
                       'Number of dimensions in higher-dimensional space must match the dimension of the given set.')
    
    if np.max(proj) > N:
        raise CORAerror('CORA:wrongValue',
                       'Specified dimensions exceed dimension of high-dimensional space.')
    
    # Call subfunction
    return S.lift_(N, proj) 