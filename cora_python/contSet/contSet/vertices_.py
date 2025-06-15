"""
vertices_ - computes the vertices of a set (internal use)

This function provides the internal implementation for vertex computation.
It should be overridden in subclasses to provide specific vertex computation logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import Any
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def vertices_(S: 'ContSet', method: str = 'convHull', *args, **kwargs) -> np.ndarray:
    """
    Computes the vertices of a set (internal use, see also contSet/vertices)
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific vertex computation logic.
    
    Args:
        S: contSet object
        method: Method for vertex computation
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        np.ndarray: Numeric matrix of vertices
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S = interval([1, 2], [3, 4])
        >>> V = vertices_(S, 'convHull')
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'vertices_ not implemented for {type(S).__name__} with method {method}') 