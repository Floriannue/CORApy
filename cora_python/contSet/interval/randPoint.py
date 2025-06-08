"""
randPoint - computes random point in interval

This function generates random points within an interval according to different
sampling strategies.

Authors: Florian Nüssel (Python implementation)
Date: 2025-01-08
"""

from typing import Union, Optional
import numpy as np


def randPoint(interval_obj, N: int = 1, type_: str = 'standard') -> np.ndarray:
    """
    Computes random point in interval
    
    Syntax:
        p = randPoint(I)
        p = randPoint(I, N)
        p = randPoint(I, N, type_)
        p = randPoint(I, 'all', 'extreme')
    
    Args:
        interval_obj: Interval object
        N: Number of random points (default: 1)
        type_: Type of random point ('standard', 'extreme') (default: 'standard')
    
    Returns:
        p: Random point(s) in interval
    
    Example:
        I = Interval([-2, 1], [3, 2])
        p = randPoint(I, 20)
    """
    
    # Handle 'all' vertices case for extreme points
    if isinstance(N, str) and N == 'all' and type_ == 'extreme':
        return interval_obj.vertices()
    
    # Get object properties
    c = interval_obj.center()
    r = interval_obj.rad()
    n = interval_obj.dim()
    
    # Generate different types of points
    if type_ == 'standard' or type_.startswith('uniform'):
        # Standard uniform sampling within interval
        if r.ndim > 1 and r.shape[1] > 1:
            if r.shape[0] > 1:
                # Both dimensions larger than 1 -> interval matrix
                raise ValueError('randPoint not defined for interval matrices!')
            else:
                # Row interval
                p = c + (-1 + 2 * np.random.rand(N, len(r))) * r
        else:
            # Column interval
            if r.ndim == 1:
                r = r.reshape(-1, 1)
            if c.ndim == 1:
                c = c.reshape(-1, 1)
            p = c + (-1 + 2 * np.random.rand(len(r), N)) * r
            
    elif type_ == 'extreme':
        # Sample from vertices
        vertices = interval_obj.vertices()
        if vertices.shape[1] == 0:
            p = vertices
        else:
            # Randomly select from vertices
            n_vertices = vertices.shape[1]
            indices = np.random.choice(n_vertices, N, replace=True)
            p = vertices[:, indices]
            
    else:
        raise ValueError(f"Unknown sampling type: {type_}")
    
    return p 