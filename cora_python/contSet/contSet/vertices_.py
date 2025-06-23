"""
vertices_ - computes the vertices of a set (internal use)

This function provides the internal implementation for vertex computation.
It should be overridden in subclasses to provide specific vertex computation logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Any
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def vertices_(S: 'ContSet', method: str = 'convHull', *args, **kwargs) -> np.ndarray:
    """
    Computes the vertices of a set (internal use, see also contSet/vertices)
    
    This function uses polymorphic dispatch to call the appropriate subclass
    implementation of vertices_, or provides the base implementation.
    
    Args:
        S: contSet object
        method: Method for vertex computation
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        np.ndarray: Numeric matrix of vertices
        
    Raises:
        CORAerror: If vertices_ is not implemented for the specific set type
        
    Example:
        >>> # This will dispatch to the appropriate subclass implementation
        >>> S = interval([1, 2], [3, 4])
        >>> V = vertices_(S, 'convHull')
    """
    # Check if subclass has overridden vertices_ method
    base_class = type(S).__bases__[0] if type(S).__bases__ else None
    if (hasattr(type(S), 'vertices_') and 
        base_class and hasattr(base_class, 'vertices_') and
        type(S).vertices_ is not base_class.vertices_):
        # Try calling with method first, fallback to no method for interval-like classes
        try:
            return type(S).vertices_(S, method, *args, **kwargs)
        except TypeError:
            # Some implementations (like interval) don't take method parameter
            return type(S).vertices_(S, *args, **kwargs)
    else:
        # Base implementation - throw error as this method should be overridden
        raise CORAerror('CORA:noops',
                       f'vertices_ not implemented for {type(S).__name__} with method {method}') 