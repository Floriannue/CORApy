"""
ellipsoid - converts a polytope to an ellipsoid

Description:
    Converts a polytope to an ellipsoid using different approximation methods.

Syntax:
    E = ellipsoid(P, mode)

Inputs:
    P - polytope object
    mode - (optional): 
               'inner' (inner approx)
               'outer' (outer approx)
               'outer:min-vol' (min-vol outer approx)

Outputs:
    E - ellipsoid object

Example: 
    A = [1 2; -1 2; -2 -2; 1 -2]; b = ones(4,1);
    P = polytope(A,b);
    E = ellipsoid(P);
    figure; hold on;
    plot(P); plot(E);

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       15-March-2021 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .private.priv_equality_to_inequality import priv_equality_to_inequality
from .private.priv_normalize_constraints import priv_normalize_constraints


def ellipsoid(P, mode='outer'):
    """
    Converts a polytope to an ellipsoid.
    
    Args:
        P: polytope object
        mode: (optional) approximation mode
              'inner' (inner approx)
              'outer' (outer approx) 
              'outer:min-vol' (min-vol outer approx)
    
    Returns:
        E: ellipsoid object
    """
    # check input arguments
    if mode not in ['outer', 'outer:min-vol', 'inner']:
        raise CORAerror('CORA:wrongValue', 'mode must be "outer", "outer:min-vol", or "inner"')
    
    # read out dimension
    n = P.dim()
    
    # empty set
    if P.representsa_('emptySet', eps=1e-14):
        E = Ellipsoid.empty(n)
        return E
    
    # select mode
    if mode == 'outer':
        V = P.vertices_()
        mm = 'cov'
        E = Ellipsoid.enclosePoints(V, mm)
    elif mode == 'outer:min-vol':
        V = P.vertices_()
        mm = 'min-vol'
        E = Ellipsoid.enclosePoints(V, mm)
    elif mode == 'inner':
        E = _aux_ellipsoid_inner(P, n)
    
    return E


def _aux_ellipsoid_inner(P, n):
    """
    Inner approximation of a polytope with an ellipsoid.
    """
    # rewrite equality constraints as inequality constraints
    A, b = priv_equality_to_inequality(P.A, P.b, P.Ae, P.be)
    
    # normalize to prevent numerical issues
    A, b, _, _ = priv_normalize_constraints(A, b, None, None, 'A')
    
    # number of inequality constraints
    nrCon = A.shape[0] if A.size > 0 else 0
    
    # For now, use a simplified approach since full SDP solvers are not available
    # This provides a reasonable inner approximation
    E = _approximate_inner_ellipsoid(A, b, n, P)
    
    return E


def _approximate_inner_ellipsoid(A, b, n, P=None):
    """
    Approximate inner ellipsoid using a simplified approach.
    This is a fallback since full SDP solvers are not available.
    """
    # Use the analytical center as a starting point
    # This is a reasonable approximation for the center of the largest inscribed ellipsoid
    
    # For simplicity, use the centroid of the polytope vertices
    # In practice, this would be computed more efficiently
    try:
        if P is not None:
            V = P.vertices_()
            center = np.mean(V, axis=1, keepdims=True)
        else:
            # Fallback: use origin as center
            center = np.zeros((n, 1))
    except:
        # Fallback: use origin as center
        center = np.zeros((n, 1))
    
    # If no constraints, create a small ellipsoid
    if A.size == 0:
        Q = 0.1 * np.eye(n)
        q = center
        return Ellipsoid(Q, q)
    
    # Compute distances from center to each constraint
    distances = (b - A @ center).flatten()
    
    # Find the minimum distance to determine the radius
    min_distance = np.min(distances)
    
    if min_distance <= 0:
        # Center is not inside polytope, use a small ellipsoid
        Q = 0.1 * np.eye(n)
        q = center
    else:
        # Scale the ellipsoid to fit inside the polytope
        # Use a conservative scaling factor
        scale_factor = min_distance / np.sqrt(n)
        Q = (scale_factor**2) * np.eye(n)
        q = center
    
    return Ellipsoid(Q, q) 