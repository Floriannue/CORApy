"""
matPolytope - converts a polytope into a matPolytope object

Syntax:
    matP = matPolytope(P)

Inputs:
    P - polytope object

Outputs:
    matP - matPolytope

Authors:       Matthias Althoff, Tobias Ladner, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       21-June-2010 (MATLAB)
Last update:   02-May-2024 (TL, moved out of constructor) (MATLAB)
               12-July-2024 (MW, fix broken function) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.matrixSet.matPolytope.matPolytope import MatPolytope

if TYPE_CHECKING:
    from .polytope import Polytope

def matPolytope(P: 'Polytope') -> 'MatPolytope':
    """
    Converts a polytope object into a matPolytope object.

    Args:
        P: Polytope object.

    Returns:
        MatPolytope object.
    """

    # Get vertices from polytope class
    V = P.vertices_()

    # Rewrite vertices in n x 1 x N format (n = dimension, N = number of vertices)
    # MATLAB: V = reshape(V,[dim(P),1,size(V,2)]);
    # Python: V is (n x num_vertices), reshape to (n, 1, num_vertices)
    
    if V.size == 0:
        # Handle empty vertices case (e.g., P.empty(n) -> V is (n,0))
        # In MATLAB, reshape(zeros(n,0),[n,1,0]) would result in (n,1,0).
        # In numpy, np.reshape(np.zeros((n,0)), (n,1,0)) works for n>0.
        # If P.dim() is 0, it would be (0,0) -> (0,1,0) (wrong if n=0)
        mat_P_dim = P.dim()
        if mat_P_dim == 0:
            V_reshaped = np.array([]).reshape(0,0,0) # Consistent empty 3D array
        else:
            V_reshaped = np.zeros((mat_P_dim, 1, 0)) # n x 1 x 0 for empty
    else:
        V_reshaped = V.reshape(P.dim(), 1, V.shape[1])

    # Init matrix polytope
    matP = MatPolytope(V_reshaped)

    return matP
