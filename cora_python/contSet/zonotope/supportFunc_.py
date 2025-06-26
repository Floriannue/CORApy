"""
supportFunc_ method for zonotope class
"""

import numpy as np
from typing import Tuple, Union, Optional
from .zonotope import Zonotope


def supportFunc_(Z: Zonotope, dir: np.ndarray, type: str = 'upper') -> Tuple[Union[float, 'Interval'], np.ndarray, np.ndarray]:
    """
    Calculates the upper or lower bound of a zonotope along a certain direction
    
    Args:
        Z: zonotope object
        dir: direction for which the bounds are calculated (vector)
        type: upper bound, lower bound, or both ('upper', 'lower', 'range')
        
    Returns:
        val: bound of the zonotope in the specified direction
        x: support vector
        fac: factor values that correspond to the bound
    """
    from cora_python.contSet.interval import Interval
    
    # Ensure dir is a column vector
    if dir.ndim == 1:
        dir = dir.reshape(-1, 1)
    
    # Zonotope is empty if and only if the center is empty
    # (same as call to representsa_(Z,'emptySet',0) but much faster...)
    if Z.c is None or Z.c.size == 0:
        x = np.array([])
        if type == 'upper':
            val = -np.inf
        elif type == 'lower':
            val = np.inf
        elif type == 'range':
            val = Interval(-np.inf, np.inf)
        fac = np.array([])
        return val, x, fac
    
    # Get object properties
    c = Z.c
    G = Z.G
    
    # Project zonotope onto the direction
    c_ = np.dot(dir.T, c).item()  # Scalar value
    G_ = np.dot(dir.T, G).flatten()  # 1D array
    
    # Upper or lower bound
    if type == 'lower':
        val = c_ - np.sum(np.abs(G_))
        fac = -np.sign(G_)
        
    elif type == 'upper':
        val = c_ + np.sum(np.abs(G_))
        fac = np.sign(G_)
        
    elif type == 'range':
        lower_val = c_ - np.sum(np.abs(G_))
        upper_val = c_ + np.sum(np.abs(G_))
        val = Interval(lower_val, upper_val)
        fac_lower = -np.sign(G_)
        fac_upper = np.sign(G_)
        fac = np.column_stack([fac_lower, fac_upper])
    else:
        raise ValueError(f"Unknown type: {type}. Must be 'upper', 'lower', or 'range'")
    
    # Compute support vector
    if type == 'range':
        # For range, we need to handle the two-column fac matrix
        x_lower = (c + G @ fac[:, 0:1]).flatten()
        x_upper = (c + G @ fac[:, 1:2]).flatten()
        x = np.column_stack([x_lower, x_upper])
    else:
        x = (c + G @ fac.reshape(-1, 1)).flatten()
    
    return val, x, fac 