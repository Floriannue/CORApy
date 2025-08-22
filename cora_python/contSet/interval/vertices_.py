"""
vertices_ - Computes vertices of an interval object

This function computes all vertices (corner points) of an interval.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 24-July-2006 (MATLAB)
Last update: 04-July-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from itertools import product
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interval import Interval

def vertices_(I: 'Interval', method: str = 'convHull', *args, **kwargs) -> np.ndarray:
    """
    Computes vertices of an interval object
    
    Args:
        I: Interval object
        method: Method for vertex computation (unused, kept for interface compatibility)
        *args: Additional arguments (unused)
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        np.ndarray: Vertices (each column is a vertex)
        
    Raises:
        CORAerror: If interval is an n-d array with n > 1
        
    Example:
        >>> I = Interval([1, -1], [2, 1])
        >>> V = vertices_(I)
    """
    # Check if not a matrix set
    n = I.dim()
    if isinstance(n, (list, tuple)) and len(n) > 1:
        raise CORAerror('CORA:wrongValue', 
                       'Interval must not be an n-d array with n > 1.')
    
    # Empty case
    if n == 0 or I.inf.size == 0:
        # Return empty array with proper shape (n_dims, 0)
        if hasattr(I, 'inf') and I.inf.size == 0:
            n_dims = I.inf.shape[0] if I.inf.ndim > 0 else 0
            return np.zeros((n_dims, 0))
        return np.array([])
    
    # Check whether there is a non-zero radius in all dimensions
    idx_zero_dim = np.abs(I.rad()) <= 1e-15  # withinTol equivalent
    
    if np.any(idx_zero_dim):
        # Remove dimensions with zero radius -> save indices and add later
        val_zero_dim = I.inf[idx_zero_dim]
        I_proj = I.project(~idx_zero_dim)
        
        if I_proj.dim() == 0:
            # All dimensions have zero radius
            return val_zero_dim.reshape(-1, 1)
        
        # Compute vertices for non-degenerate dimensions
        V_proj = I_proj.vertices_()
        
        # Add back removed dimensions
        V = np.zeros((n, V_proj.shape[1]))
        V[~idx_zero_dim, :] = V_proj
        V[idx_zero_dim, :] = np.tile(val_zero_dim.reshape(-1, 1), (1, V_proj.shape[1]))
        
        return V
    
    else:
        # No degenerate dimensions, computing full vertices
        
        if n == 1:
            # One-dimensional case - vertices should be (1, 2) shape: [inf, sup]
            V = np.array([[I.inf[0], I.sup[0]]])
            return V
        
        else:
            # Multi-dimensional case
            nr_comb = 2 ** n
            
            # Create factors - mimic MATLAB's combinator(2,n,'p','r')-1
            # This generates all combinations of 0s and 1s in a specific order
            fac = np.zeros((nr_comb, n), dtype=bool)
            for i in range(nr_comb):
                # Generate combinations in the same order as MATLAB's combinator
                for j in range(n):
                    fac[i, j] = (i // (2 ** (n - 1 - j))) % 2 == 1
            
            # Create vertices matrix - initialize with lower bounds
            V = np.tile(I.inf.flatten(), (nr_comb, 1)).T
            
            # Read out supremum
            ub = I.sup.flatten()
            
            # Fill in vertices where factor is True
            for i in range(nr_comb):
                V[fac[i], i] = ub[fac[i]]
            
            # 2D: sort vertices counter-clockwise (MATLAB: V(:,[1 2 4 3]))
            if n == 2 and V.shape[1] == 4:
                V = V[:, [0, 1, 3, 2]]  # Convert MATLAB [1 2 4 3] to Python [0 1 3 2]
            
            return V
    
    # Function ends here - removed duplicate code 