"""
sparse - Converts a zonotope (center and generator) into sparse representation

Syntax:
    Z_sparse = sparse(Z)

Inputs:
    Z - zonotope object

Outputs:
    Z_sparse - zonotope with sparse center vector and generator matrix

Example:
    from cora_python.contSet.zonotope import Zonotope, sparse
    import numpy as np
    import scipy.sparse as sp
    c = np.random.rand(10, 1)
    c[c < 0.8] = 0
    G = np.random.rand(10, 10)
    G[G < 0.8] = 0
    Z = Zonotope(c, G)
    Z_sparse = sparse(Z)
    print(Z.c, Z.G)
    print(Z_sparse.c, Z_sparse.G)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Maximilian Perschl (MATLAB)
               Python translation by AI Assistant
Written:       02-June-2025 (MATLAB)
Last update:   --- (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def sparse(Z: Zonotope) -> Zonotope:
    """
    Converts a zonotope (center and generator) into sparse representation.
    """

    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Convert to sparse representation
    c_sparse = sp.csr_matrix(Z.c)
    G_sparse = sp.csr_matrix(Z.G)
    
    # Create new zonotope with sparse matrices
    Z_sparse = Zonotope(c_sparse, G_sparse)
    
    return Z_sparse 