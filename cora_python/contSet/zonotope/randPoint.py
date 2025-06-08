"""
randPoint - generates random points within a zonotope

This function generates random points within a zonotope according to different
sampling strategies.

Authors: Florian Nüssel (Python implementation)
Date: 2025-01-08
"""

from typing import Union, Optional
import numpy as np


def randPoint(zonotope_obj, N: int = 1, type_: str = 'standard') -> np.ndarray:
    """
    Generates random points within a zonotope
    
    Syntax:
        p = randPoint(Z)
        p = randPoint(Z, N)
        p = randPoint(Z, N, type_)
        p = randPoint(Z, 'all', 'extreme')
    
    Args:
        zonotope_obj: Zonotope object
        N: Number of random points (default: 1)
        type_: Type of random point ('standard', 'extreme') (default: 'standard')
    
    Returns:
        p: Random point(s) in zonotope
    
    Example:
        Z = Zonotope([1, 0], [[1, 0, 1], [-1, 2, 1]])
        p = randPoint(Z)
    """
    
    # Handle 'all' vertices case for extreme points
    if isinstance(N, str) and N == 'all' and type_ == 'extreme':
        # For zonotopes, vertices can be computed but it's expensive
        # For now, return a reasonable number of extreme points
        return _aux_randPoint_extreme(zonotope_obj, 2**min(zonotope_obj.G.shape[1], 10))
    
    # Zonotope is just a point -> replicate center N times
    if _representsa_point(zonotope_obj):
        return np.tile(zonotope_obj.c.reshape(-1, 1), (1, N))
    
    # Generate different types of random points
    if type_ == 'standard':
        return _aux_randPoint_standard(zonotope_obj, N)
    elif type_ == 'extreme':
        return _aux_randPoint_extreme(zonotope_obj, N)
    else:
        raise ValueError(f"Unknown sampling type: {type_}")


def _representsa_point(zonotope_obj, tol: float = 1e-10) -> bool:
    """Check if zonotope represents a single point"""
    if zonotope_obj.G is None or zonotope_obj.G.size == 0:
        return True
    return np.all(np.abs(zonotope_obj.G) < tol)


def _aux_randPoint_standard(zonotope_obj, N: int) -> np.ndarray:
    """Generate standard random points within zonotope"""
    # Get zonotope properties
    c = zonotope_obj.c.reshape(-1, 1)
    G = zonotope_obj.G
    
    if G is None or G.size == 0:
        return np.tile(c, (1, N))
    
    # Generate random factors in [-1, 1]
    factors = -1 + 2 * np.random.rand(G.shape[1], N)
    
    # Compute random points: p = c + G * factors
    p = c + G @ factors
    
    return p


def _aux_randPoint_extreme(zonotope_obj, N: int) -> np.ndarray:
    """Generate extreme random points (vertices) of zonotope"""
    # Get zonotope properties
    c = zonotope_obj.c.reshape(-1, 1)
    G = zonotope_obj.G
    
    if G is None or G.size == 0:
        return np.tile(c, (1, N))
    
    # Generate random factors that are either -1 or +1 (extreme points)
    factors = 2 * np.random.randint(0, 2, size=(G.shape[1], N)) - 1
    
    # Compute extreme points: p = c + G * factors
    p = c + G @ factors
    
    return p 