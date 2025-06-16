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
from .dim import dim
from .rad import rad
from .project import project
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def vertices_(I: 'Interval') -> np.ndarray:
    """
    Computes vertices of an interval object
    
    Args:
        I: Interval object
        
    Returns:
        np.ndarray: Vertices (each column is a vertex)
        
    Raises:
        CORAError: If interval is an n-d array with n > 1
        
    Example:
        >>> I = Interval([1, -1], [2, 1])
        >>> V = vertices_(I)
    """
    # Check if not a matrix set
    n = dim(I)
    if isinstance(n, (list, tuple)) and len(n) > 1:
        raise CORAError('CORA:wrongValue', 
                       'Interval must not be an n-d array with n > 1.')
    
    # Empty case
    if n == 0 or I.inf.size == 0:
        # Return empty array with proper shape (n_dims, 0)
        if hasattr(I, 'inf') and I.inf.size == 0:
            n_dims = I.inf.shape[0] if I.inf.ndim > 0 else 0
            return np.zeros((n_dims, 0))
        return np.array([])
    
    # Check whether there is a non-zero radius in all dimensions
    idx_zero_dim = np.abs(rad(I)) <= 1e-15  # withinTol equivalent
    
    if np.any(idx_zero_dim):
        # Remove dimensions with zero radius -> save indices and add later
        val_zero_dim = I.inf[idx_zero_dim]
        I_proj = project(I, ~idx_zero_dim)
        
        if dim(I_proj) == 0:
            # All dimensions have zero radius
            return val_zero_dim.reshape(-1, 1)
        
        # Compute vertices for non-degenerate dimensions
        V_proj = vertices_(I_proj)
        
        # Add back removed dimensions
        V = np.zeros((n, V_proj.shape[1]))
        V[idx_zero_dim, :] = np.tile(val_zero_dim.reshape(-1, 1), (1, V_proj.shape[1]))
        V[~idx_zero_dim, :] = V_proj
        
    else:
        # Compute all possible combinations of lower/upper bounds
        # This is equivalent to MATLAB's combinator(2, dim(I), 'p', 'r') - 1
        n_dim = dim(I)
        combinations = list(product([0, 1], repeat=n_dim))
        fac = np.array(combinations, dtype=bool)
        nr_comb = len(combinations)
        
        # Initialize all points with lower bound
        V = np.tile(I.inf.reshape(-1, 1), (1, nr_comb))
        
        # Read out supremum
        ub = I.sup
        
        # Loop over all factors
        for i in range(nr_comb):
            V[fac[i], i] = ub[fac[i]]
    
    # 2D: sort vertices counter-clockwise
    if n == 2 and V.shape[1] == 4:
        V = V[:, [0, 1, 3, 2]]  # Reorder to counter-clockwise
    
    return V 