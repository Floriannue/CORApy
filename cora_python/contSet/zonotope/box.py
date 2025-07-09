import numpy as np

from .zonotope import Zonotope
from .empty import empty

def box(Z):
    """
    Computes an enclosing axis-aligned box in generator representation.
    
    According to CORA manual Appendix A.1, this method returns a zonotope that 
    represents the smallest axis-aligned box (interval) that encloses the given zonotope.
    
    Args:
        Z: Zonotope object
        
    Returns:
        Zonotope: Axis-aligned box zonotope
        
    Examples:
        >>> c = np.array([[1], [0]])
        >>> G = np.array([[2, -1], [4, 1]])
        >>> Z = Zonotope(c, G)
        >>> Z_box = box(Z)
        >>> # Z_box has axis-aligned generators only
    """
    
    # Handle empty zonotope
    if Z.isemptyobject():
        return empty(Z.dim())
    
    # MATLAB implementation: Z.G = diag(sum(abs(Z.G),2))
    # Sum absolute values of generators along each dimension
    delta = np.sum(np.abs(Z.G), axis=1)
    
    # Create diagonal generator matrix from the sums
    G_box = np.diag(delta)
    
    # If all radii are zero (no generators), return empty generator matrix
    if np.all(delta == 0):
        G_box = np.zeros((Z.dim(), 0))
    
    return Zonotope(Z.c, G_box) 