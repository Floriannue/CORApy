"""
filterOut - deletes parallelotopes that are covered by other parallelotopes

Syntax:
    Zrem = filterOut(Z)

Inputs:
    Z - cell array of zonotope objects

Outputs:
    Zrem - cell array of remaining zonotope objects

Example:

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       09-October-2008 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import List, Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def filterOut(Z: List[Zonotope]) -> List[Zonotope]:
    """
    Deletes parallelotopes that are covered by other parallelotopes
    
    Args:
        Z: list of zonotope objects
        
    Returns:
        List of remaining zonotope objects
        
    Example:
        Z1 = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([[0], [0]]), np.array([[2, 0], [0, 2]]))
        Zrem = filterOut([Z1, Z2])
    """
    # Check for None values
    if Z is None or len(Z) == 0:
        return []
    
    # Initialize Zrem
    Zrem = []
    
    # Sort the parallelotopes by volume
    vol = []
    for i in range(len(Z)):
        from .volume_ import volume_
        vol.append(volume_(Z[i], 'exact'))
    
    # Sort by volume (ascending)
    sorted_indices = np.argsort(vol)
    
    # Convert to halfspace representation
    P = []
    for i in range(len(Z)):
        from .polytope import polytope
        P.append(polytope(Z[i]))
    
    # Intersect parallelotopes
    for i in range(len(Z)):
        ind = sorted_indices[i]
        Pint = P[ind]
        
        for j in range(len(Z)):
            if j != ind:
                Pint = Pint.difference(P[j])
        
        # Is parallelotope empty? Check if the polytope is empty
        if not Pint.isemptyobject():
            Zrem.append(Z[ind])
        else:
            print('canceled!!')
    
    return Zrem 