"""
splitFirstGen - splits first generator, which is in direction of the vector field

Syntax:
    Znew = splitFirstGen(Z)

Inputs:
    Z - list of zonotope objects

Outputs:
    Znew - list of remaining zonotope objects

Example:
    from cora_python.contSet.zonotope import Zonotope, splitFirstGen
    import numpy as np
    Z1 = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
    Z2 = Zonotope(np.array([[1], [1]]), np.array([[1, 0], [0, 1]]))
    Z_list = [Z1, Z2]
    Znew = splitFirstGen(Z_list)
    # Znew is a list of split zonotope objects

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       09-October-2008 (MATLAB)
Last update:   14-March-2019 (sort removed) (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""
import numpy as np
from typing import List
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def splitFirstGen(Z: List[Zonotope]) -> List[Zonotope]:
    """
    Splits first generator, which is in direction of the vector field.
    """
    # Initialize Znew
    Znew = []
    
    # Split first generator
    for i in range(len(Z)):
        # Find longest generator
        G = Z[i].G
        if G is None or G.size == 0:
            continue
            
        h = []
        for j in range(G.shape[1]):
            h.append(np.linalg.norm(G[:, j:j+1].T @ G, ord=1))
        
        # Find index of longest generator
        index = np.argmax(h)
        
        # Split longest generator
        from .split import split
        Ztemp = split(Z[i], index)
        
        # Write to Znew
        Znew.extend(Ztemp)
    
    return Znew 