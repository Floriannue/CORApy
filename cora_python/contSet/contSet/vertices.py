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


def vertices(S: 'ContSet', *args, **kwargs) -> np.ndarray:
    """
    Computes the vertices of a set
    
    Args:
        S: contSet object
        *args: Method and additional arguments for specific methods
        **kwargs: Additional keyword arguments
        
    Returns:
        np.ndarray: Array of vertices (each column is a vertex)
        
    Example:
        >>> S = interval([1, 2], [3, 4])
        >>> V = vertices(S)
        >>> # V contains the corner points of the interval
    """
    # Parse input arguments and set default method
    S, method, addargs = _parse_input(S, *args, **kwargs)
    
    # Call subclass method with proper error handling
    try:
        # Always pass method parameter first; if subclass expects fewer args,
        # fall back to calling without method
        try:
            res = S.vertices_(method, *addargs)
        except TypeError:
            res = S.vertices_(*addargs)
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


def _parse_input(S: 'ContSet', *args, **kwargs) -> tuple:
    """
    Parse input arguments for vertices computation (matches MATLAB aux_parseInput)
    
    Args:
        S: contSet object
        *args: Method and additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        tuple: (S, method, additional_args)
    """
    from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
    from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
    from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED
    
    # Convert args to list for processing
    args_list = list(args)
    
    # Handle method keyword argument
    if 'method' in kwargs:
        # Insert method at the beginning of args_list
        args_list.insert(0, kwargs['method'])
        # Remove method from kwargs to avoid passing it twice
        kwargs_without_method = {k: v for k, v in kwargs.items() if k != 'method'}
        # Add remaining kwargs as additional arguments
        for key, value in kwargs_without_method.items():
            args_list.append(value)
    
    # Initialize remaining as empty list
    remaining = []
    
    # Set default method based on set type (matches MATLAB exactly)
    if hasattr(S, '__class__'):
        class_name = S.__class__.__name__
        
        if class_name == 'Polytope':
            # Uses different methods
            method_list = setDefaultValues(['lcon2vert'], args_list)
            method = method_list[0]  # Extract first element from list
            remaining = []
            if CHECKS_ENABLED():
                inputArgsCheck([
                    [S, 'att', 'Polytope'],
                    [method, 'str', ['cdd', 'lcon2vert', 'comb']]
                ])
            
        elif class_name == 'ConPolyZono':
            # 'method' is number of splits
            method_list = setDefaultValues([10], args_list)
            method = method_list[0]  # Extract first element from list
            remaining = []
            if CHECKS_ENABLED():
                inputArgsCheck([
                    [S, 'att', 'ConPolyZono'],
                    [method, 'att', 'numeric', ['scalar', 'nonnan']]
                ])
            
        elif class_name == 'ConZonotope':
            method_list = setDefaultValues(['default', 1], args_list)
            method = method_list[0]  # Extract first element from list
            numDirs = method_list[1]  # Extract second element from list
            remaining = []
            if CHECKS_ENABLED():
                inputArgsCheck([
                    [S, 'att', 'ConZonotope'],
                    [method, 'str', ['default', 'template']],
                    [numDirs, 'att', 'numeric', 'isscalar']
                ])
            # For ConZonotope, we need to pass both method and numDirs
            remaining = [method, numDirs] + remaining
            
        else:
            # General set types
            method_list = setDefaultValues(['convHull'], args_list)
            method = method_list[0]  # Extract first element from list
            remaining = []
            if CHECKS_ENABLED():
                inputArgsCheck([
                    [S, 'att', 'ContSet'],
                    [method, 'str', ['convHull', 'iterate', 'polytope']]
                ])
    else:
        # Fallback for unknown types
        method_list = setDefaultValues(['convHull'], args_list)
        method = method_list[0]  # Extract first element from list
        remaining = []
    
    return S, method, remaining 