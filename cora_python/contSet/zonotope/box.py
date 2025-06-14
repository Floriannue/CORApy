import numpy as np


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
    from .zonotope import Zonotope
    
    # Handle empty zonotope
    if Z.isemptyobject():
        return Zonotope.empty(Z.dim())
    
    # Convert to interval to get bounds, then back to zonotope
    from cora_python.contSet.interval import Interval
    I = Interval(Z)
    
    # Create axis-aligned box zonotope
    c_box = I.center()
    
    # Create diagonal generator matrix from interval radii
    n = Z.dim()
    radii = I.rad()
    G_box = np.diag(radii.flatten())
    
    # Remove zero generators (dimensions with zero radius)
    non_zero_mask = radii.flatten() > np.finfo(float).eps
    if np.any(non_zero_mask):
        G_box = G_box[:, non_zero_mask]
    else:
        G_box = np.zeros((n, 0))
    
    return Zonotope(c_box, G_box) 