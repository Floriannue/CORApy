"""
encloseMany - function for the enclosure of many zonotopes

Syntax:
    [Zenclose,rotMatrixInv] = encloseMany(Z,direction)

Inputs:
    Z - list of zonotopes to be enclosed
    direction - mean direction, in which the zonotopes to be enclosed are
                heading to

Outputs:
    Zenclose - enclosing zonotope (which is an oriented rectangular hull)
    rotMatrix - rotation matrix of the oriented rectangular hull

Example: 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: dirPolytope

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       15-January-2008 (MATLAB)
Last update:   ---
Last revision: ---
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import List, Tuple
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .zonotope import Zonotope
from ..interval.interval import Interval


def encloseMany(Z: List[Zonotope], direction: np.ndarray) -> Tuple[Zonotope, np.ndarray]:
    """
    Function for the enclosure of many zonotopes.
    
    Args:
        Z: List of zonotopes to be enclosed
        direction: Mean direction, in which the zonotopes to be enclosed are heading to
        
    Returns:
        Tuple containing:
        - Zenclose: Enclosing zonotope (which is an oriented rectangular hull)
        - rotMatrixInv: Rotation matrix of the oriented rectangular hull
        
    Raises:
        CORAerror: If inputs are invalid or computation fails
    """
    # Get dimension and original axis aligned orientation
    n = len(direction)
    orient = np.eye(n)
    
    # Replace one of the axis aligned vectors
    newGen = direction / np.linalg.norm(direction)
    
    # Retrieve most aligned generator from orient
    h = np.zeros(orient.shape[1])
    for iGen in range(orient.shape[1]):
        h[iGen] = abs(newGen.T @ orient[:, iGen] / np.linalg.norm(orient[:, iGen]))
    
    ind = np.argsort(h)
    pickedIndices = ind[:-1]
    
    rotMatrix = np.hstack([newGen.reshape(-1, 1), orient[:, pickedIndices]])
    
    # Obtain and collect vertices
    Vsum = np.zeros((n, 0))
    for i in range(len(Z)):
        Zred = Z[i].reduce('parallelpiped')
        Vnew = Zred.vertices()
        Vsum = np.hstack([Vsum, Vnew])
    
    # Compute rotation matrix
    rotMatrixInv = np.linalg.inv(rotMatrix)
    
    # Rotate vertices
    V = np.linalg.solve(rotMatrix, Vsum)
    
    # Compute enclosing interval
    I = Interval.enclosePoints(V)
    
    # Instantiate zonotope
    Z_interval = Zonotope(I)
    
    # Rotate zonotope back
    Zenclose = rotMatrix @ Z_interval
    
    return Zenclose, rotMatrixInv 