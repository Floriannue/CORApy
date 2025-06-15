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
import warnings


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
        try:
            from ...polytope.polytope import Polytope
            P = Polytope(Z)
            return P.contains_(p, 'exact', tol, 0, True, scalingToggle)
        except ImportError:
            warnings.warn("Polytope class not available, falling back to venum method")
            method = 'exact:venum'
    
    if method == 'exact:venum':
        res = np.zeros(num_points, dtype=bool)
        cert = np.ones(num_points, dtype=bool)  # venum is exact, so cert is always true
        scaling = np.full(num_points, np.inf)
        
        # Check for each point whether zonotope norm is <= 1
        from ..zonotopeNorm import zonotopeNorm
        
        for i in range(num_points):
            point_diff = p[:, i:i+1] - Z.c
            scaling[i] = zonotopeNorm(Z, point_diff)
            res[i] = scaling[i] <= 1 or abs(scaling[i] - 1) <= tol
        
        # Return scalars for single point, arrays for multiple points
        if num_points == 1:
            return res[0], cert[0], scaling[0]
        else:
            return res, cert, scaling
    
    elif method in ['exact', 'approx', 'approx:st']:
        # Choose between venum and polymax based on estimated runtime
        
        # For polymax, estimate number of facets
        from ..representsa_ import representsa_
        
        if Z.G.shape[1] >= Z.G.shape[0]:
            # Non-degenerate case
            from math import comb
            try:
                num_facets = 2 * comb(Z.G.shape[1], Z.G.shape[0] - 1)
            except (ValueError, OverflowError):
                num_facets = float('inf')  # Too many facets
        else:
            # Degenerate case - would need to compute rank
            # For simplicity, assume worst case
            num_facets = 2 ** Z.G.shape[1]
        
        # Estimate runtimes (heuristic)
        runtime_halfspace = num_facets * (Z.G.shape[0] ** 4)
        runtime_LP = ((Z.G.shape[1] + 1) ** 3.5) * num_points
        
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