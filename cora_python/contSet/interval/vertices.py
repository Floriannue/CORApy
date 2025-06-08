"""
vertices - compute vertices of an interval

Syntax:
    V = vertices(I)

Inputs:
    I - interval object

Outputs:
    V - vertices of the interval (each column is a vertex)

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np
from itertools import product


def vertices(I):
    """
    Compute vertices of an interval
    
    Args:
        I: interval object
        
    Returns:
        V: Matrix where each column is a vertex (n x 2^n for n-dim interval)
    """
    # Import here to avoid circular imports
    from .interval import interval
    
    # Handle empty interval
    if I.inf.size == 0:
        return np.zeros((0, 0))
    
    # Get dimensions
    n_dims = I.dim()
    if isinstance(n_dims, list):
        # Matrix interval - not supported for vertex computation
        raise ValueError("Vertex computation not supported for matrix intervals")
    
    # Handle 1D case
    if n_dims == 1:
        # Handle scalar vs vector intervals
        if I.inf.ndim == 0:
            # Scalar interval
            inf_val = float(I.inf)
            sup_val = float(I.sup)
        else:
            # Vector interval
            inf_val = I.inf[0]
            sup_val = I.sup[0]
            
        if np.isfinite(inf_val) and np.isfinite(sup_val):
            return np.array([[inf_val, sup_val]])
        else:
            # Unbounded interval - return inf/sup as vertices
            return np.array([[inf_val, sup_val]])
    
    # For higher dimensions, compute all combinations of bounds
    bounds = []
    for i in range(n_dims):
        bounds.append([I.inf[i], I.sup[i]])
    
    # Generate all combinations (Cartesian product)
    vertex_combinations = list(product(*bounds))
    
    # Convert to matrix format (n_dims x n_vertices)
    V = np.array(vertex_combinations).T
    
    return V


def vertices_(I):
    """
    Internal version of vertices computation
    
    Args:
        I: interval object
        
    Returns:
        V: Matrix where each column is a vertex
    """
    return vertices(I) 