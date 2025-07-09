"""
rotate method for zonotope class
"""

import numpy as np
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check import inputArgsCheck


def rotate(Z: Zonotope, dims: np.ndarray, angle: float) -> Zonotope:
    """
    Rotates a zonotope projected on two coordinates with the specified angle
    
    Args:
        Z: zonotope object
        dims: projected dimensions
        angle: rotation angle (in rad)
        
    Returns:
        Rotated zonotope object
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        Z_rotated = rotate(Z, np.array([0, 1]), np.pi/4)
    """
    # Check input arguments
    inputArgsCheck([
        [Z, 'att', 'zonotope'],
        [dims, 'att', 'numeric', ['nonnan', 'vector', 'nonnegative']],
        [angle, 'att', 'numeric', ['nonnan', 'scalar']]
    ])
    
    # Check that dims has exactly 2 elements
    if len(dims) != 2:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'dims must have exactly 2 elements for rotation')
    
    # Create rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)], 
                  [np.sin(angle), np.cos(angle)]])
    
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Create a copy of the zonotope
    Z_rotated = Zonotope(Z.c.copy(), Z.G.copy())
    
    # Rotate center and generators (we know c and G are not None after the check above)
    assert Z_rotated.c is not None and Z_rotated.G is not None
    Z_rotated.c[dims] = R @ Z_rotated.c[dims]
    Z_rotated.G[dims] = R @ Z_rotated.G[dims]
    
    return Z_rotated 