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
    
    This function delegates to the object's vertices_ method if available,
    otherwise raises an error.
    
    Args:
        S: contSet object
        method: Method for vertex computation
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        np.ndarray: Numeric matrix of vertices
        
    Raises:
        CORAError: If vertices_ is not implemented for the specific set type
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S = interval([1, 2], [3, 4])
        >>> V = vertices_(S, 'convHull')
    """
    # Check if the object has a vertices_ method and use it
    if hasattr(S, 'vertices_') and callable(getattr(S, 'vertices_')):
        # Try calling with method first, fallback to no method for interval-like classes
        try:
            return S.vertices_(method, *args, **kwargs)
        except TypeError:
            # Some implementations (like interval) don't take method parameter
            return S.vertices_(*args, **kwargs)
    
    # Fallback error
    raise CORAError('CORA:noops',
                   f'vertices_ not implemented for {type(S).__name__} with method {method}') 