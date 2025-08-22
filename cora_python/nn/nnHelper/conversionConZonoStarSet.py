"""
conversionConZonoStarSet - convert a constrained zonotope to a star set
   zonotope in the given dimension

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Niklas Kochdumper, Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import Tuple


def conversionConZonoStarSet(cZ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a constrained zonotope to a star set zonotope in the given dimension.
    
    Args:
        cZ: constrained zonotope
        
    Returns:
        c, G, C, d, l, u: star set representation
        
    See also: -
    """
    m = cZ.G.shape[1]
    
    if cZ.A is None or cZ.A.size == 0:
        c = cZ.c
        G = cZ.G
        C = np.vstack([np.eye(m), -np.eye(m)])
        d = np.ones((2 * m, 1))
        u = np.ones((m, 1))
        l = -np.ones((m, 1))
    else:
        # compute point satisfying all constraints with pseudo inverse
        p_ = np.linalg.pinv(cZ.A) @ cZ.b
        
        # compute null-space of constraints
        T = np.linalg.svd(cZ.A, full_matrices=True)[2].T
        # Get the null space (columns corresponding to zero singular values)
        rank = np.sum(np.linalg.svd(cZ.A, compute_uv=False) > 1e-10)
        T = T[:, rank:]
        
        # transform boundary constraints of the factor hypercube
        A = np.vstack([np.eye(m), -np.eye(m)])
        b = np.ones((2 * m, 1))
        C = A @ T
        d = b - A @ p_
        c = cZ.c + cZ.G @ p_
        G = cZ.G @ T
        
        # Create polytope and compute interval bounds
        from cora_python.contSet.polytope import Polytope
        
        # Create polytope from constraints
        poly = Polytope(C, d)
        
        # Compute interval bounds
        int_result = poly.interval()
        
        # Extract bounds
        l = int_result.inf.reshape(-1, 1)
        u = int_result.sup.reshape(-1, 1)
    
    return c, G, C, d, l, u
