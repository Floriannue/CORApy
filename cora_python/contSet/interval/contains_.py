"""
contains_ - determines if an interval contains a set or a point

Syntax:
    res = contains_(I, S)
    res = contains_(I, S, method)
    res = contains_(I, S, method, tol)
    res, cert = contains_(I, S, ...)
    res, cert, scaling = contains_(I, S, ...)

Inputs:
    I - interval object
    S - contSet object or single point or matrix of points
    method - method used for the containment check.
       The available options are:
           - 'exact': Checks for containment exactly.
    tol - tolerance for the containment check; the higher the
       tolerance, the more likely it is that points near the boundary of I
       will be detected as lying in I, which can be useful to counteract
       errors originating from floating point errors.
    maxEval - Currently has no effect
    certToggle - if set to 'true', cert will be computed (see below),
       otherwise cert will be set to NaN.
    scalingToggle - if set to 'true', scaling will be computed (see
       below), otherwise scaling will be set to inf.

Outputs:
    res - true/false
    cert - returns true iff the result of res could be
           verified. For example, if res=false and cert=true, I is
           guaranteed to not be contained in I, whereas if res=false and
           cert=false, nothing can be deduced (S could still be
           contained in I).
           If res=true, then cert=true.
    scaling - returns the smallest number 'scaling', such that
           scaling*(I - I.center) + I.center contains S.

Example: 
    I1 = interval([-1;-2],[2;3])
    I2 = interval([0;0],[1;1])

    contains(I1,I2)
    contains(I1,[1;2])

Authors:       Niklas Kochdumper, Mark Wetzlinger, Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       16-May-2018 (MATLAB)
Last update:   02-September-2019
               15-November-2022 (MW, return logical array for points)
               25-November-2022 (MW, rename 'contains')
               20-January-2025 (AK, added cert and scaling options)
Last revision: 27-March-2023 (MW, rename contains_)
"""

import numpy as np
from typing import Union, Tuple

from .representsa_ import representsa_
from .aux_functions import _within_tol as withinTol


def contains_(I, S, method='exact', tol=1e-12, maxEval=200, certToggle=False, scalingToggle=False, *varargin):
    """
    Determines if an interval contains a set or a point.
    
    Args:
        I: interval object
        S: contSet object or single point
        method: method used for the containment check
        tol: tolerance for the containment check
        maxEval: maximal number of iterations (currently has no effect)
        certToggle: if True, cert will be computed
        scalingToggle: if True, scaling will be computed
        
    Returns:
        tuple: (res, cert, scaling) where:
            - res: True/False containment result
            - cert: True if result could be verified
            - scaling: smallest scaling factor for containment
    """
    
    # Init result
    res = False
    
    # Set in empty set
    if representsa_(I, 'emptySet', 0):
        res = representsa_(S, 'emptySet', 0)
        cert = True
        scaling = 0 if res else np.inf
        return res, cert, scaling
    
    # Point in interval containment
    if isinstance(S, (int, float, list, tuple, np.ndarray)):
        S = np.asarray(S)
        
        # Flatten interval bounds for easier handling
        I_inf = I.inf.flatten()
        I_sup = I.sup.flatten()
        
        # Handle different input shapes
        if S.ndim == 1:
            # Single point as 1D array
            if len(S) == len(I_inf):
                # Single point with same dimension as interval
                S = S.reshape(-1, 1)
            else:
                # Multiple 1D points - reshape to column vectors
                S = S.reshape(1, -1)
        elif S.ndim > 2:
            # For multi-dimensional arrays, flatten completely to match interval structure
            # The interval is flattened, so points should be too
            original_shape = S.shape
            # Flatten all but potentially the last dimension if it represents multiple points
            if S.shape[-1] > 1 and np.prod(S.shape[:-1]) == len(I_inf):
                # Last dimension represents multiple points
                S = S.reshape(-1, S.shape[-1])
            else:
                # Flatten completely and treat as single point or multiple 1D points
                S = S.flatten()
                if len(S) == len(I_inf):
                    S = S.reshape(-1, 1)
                else:
                    # Try to infer structure - if divisible, treat as multiple points
                    if len(S) % len(I_inf) == 0:
                        n_points = len(S) // len(I_inf)
                        S = S.reshape(len(I_inf), n_points)
                    else:
                        S = S.reshape(-1, 1)
        
        n_dims, n_points = S.shape
        
        # Check dimension compatibility
        if n_dims != len(I_inf):
            raise ValueError(f"Dimension mismatch: interval has {len(I_inf)} dimensions, points have {n_dims}")
        
        if scalingToggle:
            c = (I_inf + I_sup) / 2
            if np.any(np.isnan(c)):
                nan_instances = np.where(np.isnan(c))[0]
                for i in nan_instances:
                    if I_inf[i] == -np.inf and I_sup[i] == np.inf:
                        c[i] = 0
                    else:
                        raise ValueError("Cannot determine center of the interval, as it has a dimension unbounded only in one direction")
            
            # Compute scaling for each point
            scaling = np.zeros(n_points)
            for j in range(n_points):
                point = S[:, j]
                point_centered = point - c
                I_inf_centered = I_inf - c
                I_sup_centered = I_sup - c
                
                # Compute scaling needed for each dimension
                scaling_per_dim = np.zeros(n_dims)
                for i in range(n_dims):
                    if I_inf_centered[i] == 0 and I_sup_centered[i] == 0:
                        # Interval is a point
                        scaling_per_dim[i] = 0 if point_centered[i] == 0 else np.inf
                    else:
                        # Find which bound is more restrictive
                        if point_centered[i] >= 0:
                            scaling_per_dim[i] = abs(point_centered[i] / I_sup_centered[i]) if I_sup_centered[i] != 0 else (0 if point_centered[i] == 0 else np.inf)
                        else:
                            scaling_per_dim[i] = abs(point_centered[i] / I_inf_centered[i]) if I_inf_centered[i] != 0 else (0 if point_centered[i] == 0 else np.inf)
                
                scaling[j] = np.max(scaling_per_dim)
            
            res = scaling <= 1 + tol
            cert = np.ones_like(res, dtype=bool)
        else:
            # Check if points are within bounds
            res = np.zeros(n_points, dtype=bool)
            for j in range(n_points):
                point = S[:, j]
                within_bounds = np.all((I_inf <= point + tol) & (point <= I_sup + tol))
                res[j] = within_bounds
            
            cert = np.ones_like(res, dtype=bool)
            scaling = np.zeros_like(res, dtype=float)
        
        # Return scalars for single point, arrays for multiple points
        if n_points == 1:
            return res[0], cert[0], scaling[0]
        else:
            return res, cert, scaling
    
    # Interval in interval containment
    elif hasattr(S, 'inf') and hasattr(S, 'sup'):
        # We know I is not an empty set, so only check S
        if representsa_(S, 'emptySet', 0):
            scaling = 0
            cert = True
            res = True
            return res, cert, scaling
        
        # Compute scaling?
        if scalingToggle:
            I_inf = I.inf.flatten()
            I_sup = I.sup.flatten()
            S_inf = S.inf.flatten()
            S_sup = S.sup.flatten()
            
            c = (I_inf + I_sup) / 2
            if np.any(np.isnan(c)):
                nan_instances = np.where(np.isnan(c))[0]
                for i in nan_instances:
                    if I_inf[i] == -np.inf and I_sup[i] == np.inf:
                        c[i] = 0
                    else:
                        raise ValueError("Cannot determine center of the interval, as it has a dimension unbounded only in one direction")
            
            I_inf_centered = I_inf - c
            I_sup_centered = I_sup - c
            S_inf_centered = S_inf - c
            S_sup_centered = S_sup - c
            
            scaling_inf = np.abs(S_inf_centered / I_inf_centered)
            scaling_sup = np.abs(S_sup_centered / I_sup_centered)
            
            # Need to remove NaNs, which can happen only if both coordinates are the same, and are 0 or inf
            for i in range(len(I_inf)):
                if np.isnan(scaling_inf[i]):
                    a = S_inf_centered[i]
                    b = I_inf_centered[i]
                    if b == 0:
                        scaling_inf[i] = 0
                    elif a == b:
                        scaling_inf[i] = np.inf
                    else:
                        scaling_inf[i] = 0  # This can technically never happen since I and S are both non-empty
                
                # Do the same for scaling_sup
                if np.isnan(scaling_sup[i]):
                    a = S_sup_centered[i]
                    b = I_sup_centered[i]
                    if b == 0:
                        scaling_sup[i] = 0
                    elif a == b:
                        scaling_sup[i] = np.inf
                    else:
                        scaling_sup[i] = 0
            
            scaling = np.max(np.concatenate([scaling_inf, scaling_sup]))
            cert = True
            res = scaling <= 1 + tol
            return res, cert, scaling
        
        else:  # do not compute scaling
            I_inf = I.inf.flatten()
            I_sup = I.sup.flatten()
            S_inf = S.inf.flatten()
            S_sup = S.sup.flatten()
            
            if (np.all(I_sup >= S_sup) or np.all(withinTol(I_sup, S_sup, tol))) and \
               (np.all(I_inf <= S_inf) or np.all(withinTol(I_inf, S_inf, tol))):
                res = True
                cert = True
                scaling = 0 if scalingToggle else np.inf
            else:
                res = False
                cert = True
                scaling = np.inf
            return res, cert, scaling
    
    # Other set in interval containment
    else:
        # Convert to polytope and use polytope containment
        # For now, implement a basic fallback
        try:
            # Try to get vertices of S and check if all are contained
            if hasattr(S, 'vertices_'):
                vertices = S.vertices_()
                if vertices.size > 0:
                    vertex_results, vertex_certs, vertex_scalings = contains_(I, vertices, method, tol, maxEval, certToggle, scalingToggle)
                    if isinstance(vertex_results, np.ndarray):
                        res = np.all(vertex_results)
                        cert = np.all(vertex_certs)
                        scaling = np.max(vertex_scalings) if scalingToggle else np.nan
                    else:
                        res = vertex_results
                        cert = vertex_certs
                        scaling = vertex_scalings
                    return res, cert, scaling
        except Exception:
            pass
        
        # Fallback: not implemented for this set type
        res = False
        cert = False
        scaling = np.inf
        return res, cert, scaling 
