"""
priv_compactv - private helper function for compacting vertex representations

This is a private helper function used internally by polytope methods.
"""

import numpy as np
from scipy.spatial import ConvexHull

def priv_compactv(V, tol=1e-12):
    """
    Private helper for compacting vertex representations
    
    Args:
        V: vertex matrix (n x m)
        tol: tolerance for removing redundant vertices
        
    Returns:
        V_compact: compacted vertex matrix
    """
    if V.size == 0:
        return V
    
    n, m = V.shape
    
    if m <= n + 1:
        # Not enough vertices to remove any
        return V
    
    try:
        # Compute convex hull to get extreme vertices
        hull = ConvexHull(V.T)
        V_compact = V[:, hull.vertices]
    except:
        # Fallback if convex hull fails
        V_compact = V
    
    return V_compact
