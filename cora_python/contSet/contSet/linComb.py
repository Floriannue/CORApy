"""
linComb - computes the linear combination of two sets

This function computes the linear combination of two sets:
{ λs₁ + (1-λ)s₂ | s₁ ∈ S₁, s₂ ∈ S₂, λ ∈ [0,1] }

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def linComb(S1: 'ContSet', S2: 'ContSet') -> 'ContSet':
    """
    Computes the linear combination of two sets
    
    Computes the set { λs₁ + (1-λ)s₂ | s₁ ∈ S₁, s₂ ∈ S₂, λ ∈ [0,1] }
    
    Args:
        S1: First contSet object
        S2: Second contSet object
        
    Returns:
        ContSet: Linear combination of S₁ and S₂
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([0, 1], [2, 3])
        >>> S_comb = linComb(S1, S2)
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'linComb not implemented for {type(S1).__name__} and {type(S2).__name__}') 