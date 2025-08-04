"""
box - computes an enclosing axis-aligned box; the result is equivalent to a conversion to intervals but yields a zonotope representation

Syntax:
    Z = box(Z)

Inputs:
    Z - zonotope object

Outputs:
    Z - zonotope object

Example:
    from cora_python.contSet.zonotope import Zonotope, box
    import numpy as np
    Z = Zonotope(np.array([[1], [-1]]), np.array([[-3, 2, 1], [-1, 0, 3]]))
    B = box(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       09-March-2009 (MATLAB)
Last update:   27-August-2019 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""
import numpy as np
from .zonotope import Zonotope
from .empty import empty

def box(Z):
    """
    Computes an enclosing axis-aligned box.
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