"""
priv_cartprod - private helper function for Cartesian product operations

This is a private helper function used internally by polytope methods.
"""

import numpy as np

def priv_cartprod(*args):
    """
    Private helper for Cartesian product operations
    
    Args:
        *args: arrays or lists to compute Cartesian product of
        
    Returns:
        cartesian_product: array of all combinations
    """
    if len(args) == 0:
        return np.array([])
    
    # Convert all inputs to numpy arrays
    arrays = [np.asarray(arg) for arg in args]
    
    # Handle empty arrays
    if any(arr.size == 0 for arr in arrays):
        return np.array([])
    
    # Compute Cartesian product
    from itertools import product
    cartesian_product = np.array(list(product(*arrays)))
    
    return cartesian_product
