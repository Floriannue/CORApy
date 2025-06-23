"""
randPoint_ - generates a random point within a given continuous set (internal use)

This function provides the internal implementation for random point generation.
It should be overridden in subclasses to provide specific random point logic.

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-November-2020 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Union
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def randPoint_(S: 'ContSet', N: int = 1, method: str = 'standard') -> np.ndarray:
    """
    Generates random points inside a set (internal use)
    
    This function uses polymorphic dispatch to call the appropriate subclass
    implementation of randPoint_, or provides the base implementation.
    
    Args:
        S: contSet object
        N: number of points to generate
        method: method for point generation
        
    Returns:
        np.ndarray: Random points
        
    Raises:
        CORAerror: If randPoint_ is not implemented for the specific set type
    """
    # Check if subclass has overridden randPoint_ method
    base_class = type(S).__bases__[0] if type(S).__bases__ else None
    if (hasattr(type(S), 'randPoint_') and 
        base_class and hasattr(base_class, 'randPoint_') and
        type(S).randPoint_ is not base_class.randPoint_):
        return type(S).randPoint_(S, N, method)
    else:
        # Base implementation - throw error as this method should be overridden
        raise CORAerror("CORA:noops", f"Function randPoint_ not implemented for class {type(S).__name__}") 