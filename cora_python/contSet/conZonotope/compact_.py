"""
compact_ - returns equal constrained zonotope in minimal representation
   1) delete all constraints of the form 0 * beta = 0 as they are
   trivially true for all values of beta; additionally, remove all
   generators which have zero-length and no corresponding entries in the
   constraint matrix

Syntax:
    cZ = compact_(cZ, method, tol)

Inputs:
    cZ - conZonotope object
    method - method for redundancy removal
             'zeros' (default): delete all constraints of the form
                  0 * beta = 0 as they are trivially true for all values
                  of beta; additionally, remove all generators which have
                  zero-length and no corresponding entries in the
                  constraint matrix
    tol - tolerance

Outputs:
    cZ - conZonotope object

References:
  [1] J. Scott et al. "Constrained zonotope: A new tool for set-based
      estimation and fault detection"

Authors:       Niklas Kochdumper, Mark Wetzlinger (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       29-July-2023 (MATLAB)
Last update:   ---
Last revision: ---
"""

import numpy as np


def compact_(cZ, method: str = 'zeros', tol: float = None):
    """
    Return equal constrained zonotope in minimal representation.
    """
    if tol is None:
        tol = float(np.finfo(float).eps)

    if method == 'all':
        return compact_(cZ, 'zeros', tol)

    if method != 'zeros':
        raise ValueError(f"Unknown method: {method}. Use 'zeros' or 'all'.")

    # constraints that are trivially true: 0 * beta = 0
    if cZ.A.size > 0 or cZ.b.size > 0:
        Ab = np.hstack([cZ.A, cZ.b]) if cZ.A.size > 0 else cZ.b
        idx_rows = ~np.any(Ab, axis=1)
        if np.any(idx_rows):
            cZ.A = cZ.A[~idx_rows, :] if cZ.A.size > 0 else cZ.A
            cZ.b = cZ.b[~idx_rows, :] if cZ.b.size > 0 else cZ.b

    # zero-length generators (corresponding columns in constraint matrix A need
    # to be all-zero as well)
    if cZ.G.size > 0:
        if cZ.A.size > 0:
            GA = np.vstack([cZ.G, cZ.A])
        else:
            GA = cZ.G
        idx_cols = ~np.any(GA, axis=0)
        if np.any(idx_cols):
            cZ.G = cZ.G[:, ~idx_cols]
            if cZ.A.size > 0:
                cZ.A = cZ.A[:, ~idx_cols]

    return cZ
