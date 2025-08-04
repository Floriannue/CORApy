"""
vertices - computes the vertices of a set

This function computes the vertices of a contSet object using various methods.
It handles different set types and provides appropriate error handling.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 18-August-2022 (MATLAB)
Last update: 12-July-2024 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Optional, Union, List, Any
import numpy as np

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def vertices(S: 'ContSet', method: Optional[str] = None, *args, **kwargs) -> np.ndarray:
    """
    Computes the vertices of a set
    
    Args:
        S: contSet object
        method: Method for computation of vertices (optional)
        *args: Additional arguments for specific methods
        **kwargs: Additional keyword arguments
        
    Returns:
        np.ndarray: Array of vertices (each column is a vertex)
        
    Example:
        >>> S = interval([1, 2], [3, 4])
        >>> V = vertices(S)
        >>> # V contains the corner points of the interval
    """
    # Parse input arguments and set default method
    S, method, addargs = _parse_input(S, method, *args, **kwargs)
    
    try:
        # Call subclass method
        res = S.vertices_(*addargs) # Removed 'method' from arguments
    except Exception as ME:
        # Catch empty set case
        if S.representsa_('emptySet', 1e-15):
            res = np.array([])
        else:
            raise ME
    
    if res.size == 0:
        # Create res with proper dimensions
        res = np.zeros((S.dim(), 0))
    
    return res


def _parse_input(S: 'ContSet', method: Optional[str] = None, *args, **kwargs) -> tuple:
    """
    Parse input arguments for vertices computation
    
    Args:
        S: contSet object
        method: Method for computation
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        tuple: (S, method, additional_args)
    """
    # Set default method based on set type
    if method is None:
        if hasattr(S, '__class__') and S.__class__.__name__ == 'Polytope':
            method = 'lcon2vert'
        elif hasattr(S, '__class__') and S.__class__.__name__ == 'ConPolyZono':
            method = 10  # number of splits
        elif hasattr(S, '__class__') and S.__class__.__name__ == 'ConZonotope':
            method = 'default'
        else:
            method = 'convHull'
    
    # Validate method based on set type
    if hasattr(S, '__class__'):
        class_name = S.__class__.__name__
        if class_name == 'Polytope':
            if method not in ['cdd', 'lcon2vert']:
                raise ValueError(f"Invalid method '{method}' for Polytope. Use 'cdd' or 'lcon2vert'.")
        elif class_name == 'ConPolyZono':
            if not isinstance(method, (int, float)) or method <= 0:
                raise ValueError(f"Method for ConPolyZono must be a positive number (splits), got {method}")
        elif class_name == 'ConZonotope':
            if method not in ['default', 'template']:
                raise ValueError(f"Invalid method '{method}' for ConZonotope. Use 'default' or 'template'.")
        else:
            if method not in ['convHull', 'iterate', 'polytope']:
                raise ValueError(f"Invalid method '{method}'. Use 'convHull', 'iterate', or 'polytope'.")
    
    return S, method, args 