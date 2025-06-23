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

# Removed static import - use object method instead
from cora_python.g.functions.matlab.validate.check import withinTol


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
    if I.representsa_('emptySet', 0):
        # Empty interval logic: empty contains empty, but not non-empty
        if hasattr(S, 'representsa_') and S.representsa_('emptySet', 0):
            # Empty interval contains empty interval
            res = True
            cert = True
            scaling = 0
        elif isinstance(S, (list, tuple, np.ndarray)):
            S_arr = np.asarray(S)
            if S_arr.size == 0:
                # Empty interval contains empty arrays
                res = True
                cert = True
                scaling = 0
            else:
                # Empty interval does not contain non-empty points
                res = False
                cert = True
                scaling = np.inf
        else:
            # Empty interval does not contain other objects
            res = False
            cert = True
            scaling = np.inf
        return res, cert, scaling
    
    # Point in interval containment
    if isinstance(S, (int, float, list, tuple, np.ndarray)):
        S = np.asarray(S)
        
        if scalingToggle:
            # Get center of interval
            c = (I.inf + I.sup) / 2
            if np.any(np.isnan(c)):
                nan_instances = np.where(np.isnan(c))
                for idx in zip(*nan_instances):
                    if I.inf[idx] == -np.inf and I.sup[idx] == np.inf:
                        c[idx] = 0
                    else:
                        raise ValueError("Cannot determine center of the interval, as it has a dimension unbounded only in one direction")
            
            # Check if this is a multi-dimensional case or regular point array case
            if S.ndim == len(I.inf.shape) + 1:
                # Multi-dimensional case: S has shape (..., n_points) where ... matches I.inf.shape
                # Expand c to match S dimensions
                c_expanded = np.expand_dims(c, axis=-1)
                
                # Compute scaling: scaling = abs(S) / abs([I.inf-c, I.sup-c])
                # This follows MATLAB: scaling = abs(S)./(abs([I.inf-c I.sup-c]));
                I_bounds = np.stack([np.expand_dims(I.inf - c, axis=-1), np.expand_dims(I.sup - c, axis=-1)], axis=-1)
                I_bounds_abs = np.abs(I_bounds)
                
                # For each point in S, compute scaling
                S_abs = np.abs(S)
                # Expand S to match I_bounds for division
                S_expanded = np.expand_dims(S_abs, axis=-1)  # Add dimension for bounds
                scaling = S_expanded / I_bounds_abs
                
                # Take maximum over bounds (inf vs sup) and then over spatial dimensions
                scaling = np.max(scaling, axis=-1)  # Max over bounds
                
                # Apply max() over spatial dimensions (equivalent to MATLAB's 1:numel(dim(I)))
                # This reduces all spatial dimensions, leaving only the point dimension
                spatial_dims = tuple(range(len(I.inf.shape)))
                if len(spatial_dims) > 0:
                    scaling = np.max(scaling, axis=spatial_dims)
                
            elif S.ndim == len(I.inf.shape) and S.shape[:-1] == I.inf.shape[:-1]:
                # Regular point array case: S has shape (n_dims, n_points), I has shape (n_dims, 1)
                n_points = S.shape[-1]
                scaling = np.zeros(n_points)
                
                for i in range(n_points):
                    point = S[..., i:i+1]  # Keep same shape as I for broadcasting
                    # Compute scaling for this point following MATLAB logic
                    point_scaling = np.abs(point) / np.abs(np.stack([I.inf - c, I.sup - c], axis=-1))
                    scaling[i] = np.max(point_scaling)
                
            else:
                # S has same dimensions as I - single point case
                # Compute scaling: scaling = abs(S) / abs([I.inf-c, I.sup-c])
                # This follows MATLAB: scaling = abs(S)./(abs([I.inf-c I.sup-c]));
                I_bounds = np.stack([I.inf - c, I.sup - c], axis=-1)  # Stack bounds along last axis
                I_bounds_abs = np.abs(I_bounds)
                
                # For each point in S, compute scaling
                S_abs = np.abs(S)
                scaling = S_abs / I_bounds_abs
                
                # Take maximum over bounds (inf vs sup) and then over spatial dimensions
                scaling = np.max(scaling, axis=-1)  # Max over bounds
                
                # Apply all() over spatial dimensions (equivalent to MATLAB's 1:numel(dim(I)))
                # This reduces all spatial dimensions, leaving only the point dimension
                spatial_dims = tuple(range(len(I.inf.shape)))
                if len(spatial_dims) > 0:
                    scaling = np.max(scaling, axis=spatial_dims)
            
            res = scaling <= 1 + tol
            cert = np.ones_like(res, dtype=bool)
            
            # Reshape to [1, n_points] to match MATLAB behavior
            res = res.reshape(1, -1)
            cert = cert.reshape(1, -1)
            scaling = scaling.reshape(1, -1)
            
        else:
            # Check if this is a multi-dimensional case (S has more dimensions than I)
            # or a regular point array case (S has same dimensions as I but different shape)
            if S.ndim == len(I.inf.shape) + 1:
                # Multi-dimensional case: S has shape (..., n_points) where ... matches I.inf.shape
                # We need to broadcast I.inf and I.sup to match S for element-wise comparison
                I_inf_expanded = np.expand_dims(I.inf, axis=-1)  # Add last dimension
                I_sup_expanded = np.expand_dims(I.sup, axis=-1)  # Add last dimension
                
                # Check containment: (I.inf < S + tol | withinTol(I.inf,S,tol)) & (I.sup > S - tol | withinTol(I.sup,S,tol))
                lower_check = (I_inf_expanded < S + tol) | withinTol(I_inf_expanded, S, tol)
                upper_check = (I_sup_expanded > S - tol) | withinTol(I_sup_expanded, S, tol)
                
                # Combine checks
                containment_check = lower_check & upper_check
                
                # Apply all() over spatial dimensions (equivalent to MATLAB's 1:numel(dim(I)))
                # This reduces all spatial dimensions, leaving only the point dimension
                spatial_dims = tuple(range(len(I.inf.shape)))
                if len(spatial_dims) > 0:
                    res = np.all(containment_check, axis=spatial_dims)
                else:
                    res = containment_check
                
                # Reshape to [1, n_points] to match MATLAB behavior: res = reshape(res,1,[]);
                res = res.reshape(1, -1)
                cert = np.ones_like(res, dtype=bool)
                scaling = np.zeros_like(res, dtype=float)
                
            elif S.ndim == len(I.inf.shape) and S.shape != I.inf.shape:
                # Regular point array case: S has shape (n_dims, n_points), I has shape (n_dims, 1)
                # Check containment for each point
                n_points = S.shape[-1]
                res = np.zeros(n_points, dtype=bool)
                
                for i in range(n_points):
                    point = S[..., i:i+1]  # Keep same shape as I for broadcasting
                    lower_check = (I.inf < point + tol) | withinTol(I.inf, point, tol)
                    upper_check = (I.sup > point - tol) | withinTol(I.sup, point, tol)
                    containment_check = lower_check & upper_check
                    res[i] = np.all(containment_check)
                
                cert = np.ones_like(res, dtype=bool)
                scaling = np.zeros_like(res, dtype=float)
                
            else:
                # S has same dimensions as I or different structure - single point case
                # Check containment: (I.inf < S + tol | withinTol(I.inf,S,tol)) & (I.sup > S - tol | withinTol(I.sup,S,tol))
                lower_check = (I.inf < S + tol) | withinTol(I.inf, S, tol)
                upper_check = (I.sup > S - tol) | withinTol(I.sup, S, tol)
                
                # Combine checks
                containment_check = lower_check & upper_check
                
                # For single point, check all dimensions
                res = np.all(containment_check)
                
                cert = True
                scaling = 0.0
        
        # Return scalars for single point, arrays for multiple points
        if np.isscalar(res) or (hasattr(res, 'size') and res.size == 1):
            # Single point case - ensure we return scalars
            res_scalar = res.item() if hasattr(res, 'item') else res
            cert_scalar = cert.item() if hasattr(cert, 'item') else cert
            scaling_scalar = scaling.item() if hasattr(scaling, 'item') else scaling
            return res_scalar, cert_scalar, scaling_scalar
        else:
            # For multi-dimensional case (S.ndim > I.ndim), preserve shape (1, n_points)
            # For regular point arrays, return as 1D arrays
            if S.ndim == len(I.inf.shape) + 1:
                # Multi-dimensional case: preserve (1, n_points) shape
                return res, cert, scaling
            else:
                # Regular point array case: return as 1D arrays
                return res.flatten(), cert.flatten(), scaling.flatten()
    
    # Interval in interval containment
    elif hasattr(S, 'inf') and hasattr(S, 'sup'):
        # We know I is not an empty set, so only check S
        if S.representsa_('emptySet', 0):
            scaling = 0
            cert = True
            res = True
            return res, cert, scaling
        
        # Compute scaling?
        if scalingToggle:
            c = (I.inf + I.sup) / 2
            if np.any(np.isnan(c)):
                nan_instances = np.where(np.isnan(c))
                for idx in zip(*nan_instances):
                    if I.inf[idx] == -np.inf and I.sup[idx] == np.inf:
                        c[idx] = 0
                    else:
                        raise ValueError("Cannot determine center of the interval, as it has a dimension unbounded only in one direction")
            
            I_centered = I.inf - c, I.sup - c
            S_centered = S.inf - c, S.sup - c
            
            scaling_inf = np.abs(S_centered[0] / I_centered[0])
            scaling_sup = np.abs(S_centered[1] / I_centered[1])
            
            # Need to remove NaNs, which can happen only if both coordinates are the same, and are 0 or inf
            # Handle NaNs element-wise
            nan_mask_inf = np.isnan(scaling_inf)
            nan_mask_sup = np.isnan(scaling_sup)
            
            if np.any(nan_mask_inf):
                for idx in zip(*np.where(nan_mask_inf)):
                    a = S_centered[0][idx]
                    b = I_centered[0][idx]
                    if b == 0:
                        scaling_inf[idx] = 0
                    elif a == b:
                        scaling_inf[idx] = np.inf
                    else:
                        scaling_inf[idx] = 0  # This can technically never happen since I and S are both non-empty
            
            if np.any(nan_mask_sup):
                for idx in zip(*np.where(nan_mask_sup)):
                    a = S_centered[1][idx]
                    b = I_centered[1][idx]
                    if b == 0:
                        scaling_sup[idx] = 0
                    elif a == b:
                        scaling_sup[idx] = np.inf
                    else:
                        scaling_sup[idx] = 0
            
            scaling = np.max([np.max(scaling_inf), np.max(scaling_sup)])
            cert = True
            res = scaling <= 1 + tol
            return res, cert, scaling
        
        else:  # do not compute scaling
            if (np.all(I.sup >= S.sup) or np.all(withinTol(I.sup, S.sup, tol))) and \
               (np.all(I.inf <= S.inf) or np.all(withinTol(I.inf, S.inf, tol))):
                res = True
                cert = True
                scaling = np.nan
            else:
                res = False
                cert = True
                scaling = np.nan
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
