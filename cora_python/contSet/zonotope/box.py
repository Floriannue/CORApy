import numpy as np

from .zonotope import Zonotope

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
        return Zonotope.empty(Z.dim())
    
    # Compute bounds directly from zonotope as in MATLAB version
    # delta = sum(abs(Z.G),2) - sum absolute values of generators per dimension
    c = Z.c
    delta = np.sum(np.abs(Z.G), axis=1, keepdims=True)
    
    # Create axis-aligned box zonotope
    c_box = c
    
    # Create diagonal generator matrix from radii (half of delta)
    n = Z.dim()
    radii = delta
    G_box = np.diag(radii.flatten())
    
    # Remove zero generators (dimensions with zero radius)
    non_zero_mask = radii.flatten() > np.finfo(float).eps
    if np.any(non_zero_mask):
        G_box = G_box[:, non_zero_mask]
    else:
        G_box = np.zeros((n, 0))
    
    return Zonotope(c_box, G_box) 