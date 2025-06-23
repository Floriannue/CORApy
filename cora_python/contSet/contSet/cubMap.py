"""
cubMap - computes the cubic map of a set

This function computes the cubic map of sets:
{ x | x_i = Σⱼ s_j (sᵀ T_{i,j} s), s ∈ S, i = 1...w }
where T_{i,j} ∈ ℝⁿˣⁿ

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, List, Union, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def cubMap(S1: 'ContSet', 
           S2: Optional['ContSet'] = None, 
           S3: Optional['ContSet'] = None,
           T: Optional[np.ndarray] = None,
           ind: Optional[List] = None) -> 'ContSet':
    """
    Computes the cubic map of a set
    
    Calculates the following set:
    { x | x_i = Σⱼ s_j (sᵀ T_{i,j} s), s ∈ S, i = 1...w }
    where T_{i,j} ∈ ℝⁿˣⁿ
    
    Args:
        S1: First contSet object
        S2: Second contSet object (optional)
        S3: Third contSet object (optional)
        T: Third-order tensor
        ind: List containing the non-zero indices of the tensor
        
    Returns:
        ContSet: Cubically mapped set
        
    Raises:
        CORAerror: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S1 = interval([1, 2], [3, 4])
        >>> T = np.random.rand(2, 2, 2)
        >>> S_cub = cubMap(S1, T=T)
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops',
                   f'cubMap not implemented for {type(S1).__name__}') 