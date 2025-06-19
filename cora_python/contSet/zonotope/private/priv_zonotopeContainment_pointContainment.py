"""
priv_zonotopeContainment_pointContainment - Checks whether a point (or point cloud) is
    contained in a zonotope.

Syntax:
    res, cert, scaling = priv_zonotopeContainment_pointContainment(Z, p, method, tol, scalingToggle)

Inputs:
    Z - zonotope object
    p - point, or matrix of points
    method - method used for the containment check.
       The available options are:
           - 'exact': Checks for containment by using either
               'exact:venum' or 'exact:polymax', depending on the number 
               of generators of Z and the size of p.
           - 'exact:venum': Checks for containment by evaluating the 
              Z-norm of p (see Algorithm 1 in [1]).
           - 'exact:polymax': Checks for containment by maximizing the
               polyhedral norm w.r.t. Z over p (see Algorithm 2 in [1]).
    tol - tolerance for the containment check; the higher the
       tolerance, the more likely it is that points near the boundary of Z
       will be detected as lying in Z, which can be useful to counteract
       errors originating from floating point errors.
    scalingToggle - if set to 'true', scaling will be computed (see
       below).

Outputs:
    res - true/false
    cert - returns true iff the result of res could be
           verified. For example, if res=false and cert=true, p is
           guaranteed to not be contained in Z, whereas if res=false and
           cert=false, nothing can be deduced (p could still be
           contained in Z).
           If res=true, then cert=true.
    scaling - returns the smallest number 'scaling', such that
           scaling*(Z - center(Z)) + center(Z) contains p.
           For priv_zonotopeContainment_pointContainment, this is an exact value.
           Note that computing this scaling factor may significantly
           increase the runtime.

References:
    [1] A. Kulmburg, M. Althoff.: On the co-NP-Completeness of the
        Zonotope Containment Problem, European Journal of Control 2021

Authors:       Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       05-February-2024 (MATLAB)
Last update:   ---
Last revision: ---
"""

import numpy as np
from typing import Tuple
from scipy.special import comb
from numpy.linalg import matrix_rank


def priv_zonotopeContainment_pointContainment(Z, p: np.ndarray, method: str = 'exact',
                                            tol: float = 1e-12, scalingToggle: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Checks whether a point (or point cloud) is contained in a zonotope.
    
    Args:
        Z: zonotope object
        p: point or matrix of points (n x m)
        method: containment check method
        tol: tolerance for containment check
        scalingToggle: if True, compute scaling factors
        
    Returns:
        tuple: (res, cert, scaling) arrays
    """

    # Ensure p is 2D
    if p.ndim == 1:
        p = p.reshape(-1, 1)

    n, num_points = p.shape

    if method == 'exact:polymax':
        # Transform Z into a polytope and do containment check there
        from ...polytope import Polytope
        P = Polytope(Z)
        res, cert, scaling = P.constraints().contains_(p, 'exact', tol, 0, True, scalingToggle)
        return res, cert, scaling

    elif method == 'exact:venum':
        res = np.zeros(num_points, dtype=bool)
        cert = np.ones(num_points, dtype=bool)  # venum is exact, so cert is always true
        scaling = np.full(num_points, np.inf)

        # Check for each point whether zonotope norm is <= 1
        from ..zonotopeNorm import zonotopeNorm
        from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

        for i in range(num_points):
            point_diff = p[:, i:i + 1] - Z.c
            scaling[i], _ = zonotopeNorm(Z, point_diff)
            res[i] = scaling[i] <= 1 or withinTol(scaling[i], 1, tol)

        # Return scalars for single point, arrays for multiple points
        if num_points == 1:
            return res[0], cert[0], scaling[0]
        else:
            return res, cert, scaling

    elif method == 'exact':
        # Choose between venum and polymax based on estimated runtime
        
        # For polymax, estimate number of facets
        from ..representsa_ import representsa_

        num_gens = Z.G.shape[1]
        dim = Z.G.shape[0]

        if num_gens >= dim:
            # Non-degenerate case
            try:
                num_facets = 2 * comb(num_gens, dim - 1, exact=True)
            except (ValueError, OverflowError):
                num_facets = float('inf')
        else:
            # Degenerate case
            try:
                rank = matrix_rank(Z.G)
                num_facets = 2 * comb(num_gens, rank - 1, exact=True)
            except (ValueError, OverflowError):
                num_facets = float('inf')

        # Estimate runtimes (heuristic from MATLAB)
        runtime_halfspace = num_facets * (dim ** 4)
        runtime_LP = ((num_gens + 1) ** 3.5) * num_points

        # Check if zonotope is a parallelotope (always prefer halfspace method)
        is_parallelotope = representsa_(Z, 'parallelotope', 1e-15)

        # Decide which method to use
        use_LP = num_facets > 100000 and runtime_halfspace > runtime_LP

        if is_parallelotope or not use_LP:
            # Use polymax method
            return priv_zonotopeContainment_pointContainment(Z, p, 'exact:polymax', tol, scalingToggle)
        else:
            # Use venum method
            return priv_zonotopeContainment_pointContainment(Z, p, 'exact:venum', tol, scalingToggle)
    else:
        raise ValueError(f"Unknown method: {method}")
