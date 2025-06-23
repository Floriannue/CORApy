"""
contains_ - determines if a capsule contains a set or a point

Syntax:
    res, cert, scaling = contains_(C, S, method, tol, maxEval, certToggle, scalingToggle)

Inputs:
    C - capsule object
    S - contSet object or single point
    method - method used for the containment check.
       Currently, the only available options are 'exact' and 'approx'.
    tol - tolerance for the containment check; the higher the
       tolerance, the more likely it is that points near the boundary of C
       will be detected as lying in C, which can be useful to counteract
       errors originating from floating point errors.
    maxEval - Currently has no effect
    certToggle - if set to True, cert will be computed (see below),
       otherwise cert will be set to NaN.
    scalingToggle - if set to True, scaling will be computed (see
       below), otherwise scaling will be set to inf.

Outputs:
    res - true/false
    cert - returns true iff the result of res could be
           verified. For example, if res=false and cert=true, S is
           guaranteed to not be contained in C, whereas if res=false and
           cert=false, nothing can be deduced (S could still be
           contained in C).
           If res=true, then cert=true.
    scaling - returns the smallest number 'scaling', such that
           scaling*(C - C.center) + C.center contains S.

Authors: Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 20-November-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.capsule.capsule import Capsule
    from cora_python.contSet.contSet.contSet import ContSet


def contains_(C: 'Capsule', S: 'ContSet', method: str = 'exact', tol: float = 1e-12, maxEval: int = 200, certToggle: bool = False, scalingToggle: bool = False):
    """
    Determines if a capsule contains a set or a point (internal method)
    
    Args:
        C: Capsule object
        S: contSet object or numerical vector
        method: method used for the containment check
        tol: tolerance for the containment check
        maxEval: maximum number of evaluations
        certToggle: if True, cert will be computed
        scalingToggle: if True, scaling will be computed
        
    Returns:
        tuple: (res, cert, scaling) where:
            - res: True/False or array of bools for multiple points
            - cert: certificate
            - scaling: scaling factor
    """
    # Initialize result
    res = True
    cert = np.nan
    scaling = np.inf
    
    if scalingToggle:
        raise CORAerror('CORA:notSupported',
                       "The computation of the scaling factor for capsules is not yet implemented.")
    
    # Point in capsule containment
    if isinstance(S, (np.ndarray, list)) and np.isrealobj(S):
        S = np.asarray(S)
        
        if S.ndim == 1:
            # Single point
            res = _aux_contains_point(C, S)
            cert = True
        else:
            # Multiple points (columns)
            res = np.zeros(S.shape[1], dtype=bool)
            cert = np.ones(S.shape[1], dtype=bool)
            
            for i in range(S.shape[1]):
                res[i] = _aux_contains_point(C, S[:, i])
        
        return res, cert, scaling
    
    # Capsule is empty
    if C.representsa_('emptySet'):
        res = S.representsa_('emptySet') if hasattr(S, 'representsa_') else False
        cert = True
        if res:
            scaling = 0
        else:
            scaling = np.inf
        return res, cert, scaling
    
    # Empty set is trivially contained
    if (hasattr(S, '__class__') and S.__class__.__name__ == 'EmptySet') or \
       (hasattr(S, 'representsa_') and S.representsa_('emptySet')):
        res = True
        cert = True
        scaling = 0
        return res, cert, scaling
    
    # Fullspace is trivially not contained
    if hasattr(S, '__class__') and S.__class__.__name__ == 'Fullspace':
        res = False
        cert = True
        scaling = np.inf
        return res, cert, scaling
    
    # Capsule in capsule containment
    if hasattr(S, '__class__') and S.__class__.__name__ == 'Capsule':
        res = _aux_contains_capsule(C, S)
        cert = True
        return res, cert, scaling
    
    # Polytopic set in capsule containment
    if hasattr(S, '__class__') and S.__class__.__name__ in [
        'ConZonotope', 'Interval', 'Polytope', 'ZonoBundle', 'Zonotope', 'Polygon'
    ]:
        # Compute vertices
        V = S.vertices()
        
        # Check containment for each vertex
        for i in range(V.shape[1]):
            if not _aux_contains_point(C, V[:, i]):
                res = False
                cert = True
                return res, cert, scaling
        
        cert = True
        return res, cert, scaling
    
    # Non-polytopic set in capsule containment
    if hasattr(S, '__class__') and S.__class__.__name__ in [
        'Ellipsoid', 'Taylm', 'PolyZonotope', 'ConPolyZono', 'SpectraShadow'
    ]:
        # If the user wants the exact result, throw an error
        if method != 'approx':
            raise CORAerror('CORA:noExactAlg', f"No exact algorithm implemented for containment check between {C.__class__.__name__} and {S.__class__.__name__}")
        
        # Check containment with over-approximating zonotope
        Z = S.zonotope()  # Convert to zonotope
        res, cert_inner, scaling_inner = contains_(C, Z, 'exact', tol, maxEval, certToggle, scalingToggle)
        
        # This evaluation is not necessarily exact
        if res:
            cert = True
        else:
            cert = False
        
        return res, cert, scaling
    
    else:
        raise CORAerror('CORA:noops', f"Operation not supported between {C.__class__.__name__} and {type(S).__name__}")


def _aux_contains_capsule(C1, C2):
    """
    Checks if capsule C2 is contained in capsule C1
    
    Args:
        C1: First capsule
        C2: Second capsule
        
    Returns:
        bool: True if C2 is contained in C1
    """
    if C2.g is None or np.allclose(C2.g, 0):
        return _aux_contains_sphere(C1, C2.c, C2.r)
    else:
        res1 = _aux_contains_sphere(C1, C2.c + C2.g, C2.r)
        res2 = _aux_contains_sphere(C1, C2.c - C2.g, C2.r)
        return res1 and res2


def _aux_contains_sphere(C, c, r):
    """
    Checks if the capsule contains the hypersphere defined by center and radius
    
    Args:
        C: Capsule object
        c: Center of sphere
        r: Radius of sphere
        
    Returns:
        bool: True if sphere is contained in capsule
    """
    # Ensure c is a column vector
    c = np.asarray(c).reshape(-1, 1) if np.asarray(c).ndim == 1 else np.asarray(c)
    
    # Check case where capsule is a hypersphere (no generator)
    if C.g is None or np.allclose(C.g, 0):
        tmp = np.linalg.norm(C.c - c) + r
        return tmp < C.r or withinTol(tmp, C.r)
    else:
        ng = np.linalg.norm(C.g)
        g_ = C.g / ng
        proj = np.dot((c - C.c).T, g_).item()  # Use .item() to get scalar
        
        # Check if center is in hypercylinder
        if abs(proj) < np.linalg.norm(C.g):
            # Compute distance to axis
            diff = (C.c + proj * g_) - c
            
            # Check if distance to axis is smaller than the radius
            tmp = np.linalg.norm(diff) + r
            return tmp < C.r or withinTol(tmp, C.r)
        else:
            # Check if point is in upper or lower hypersphere
            tmp = np.linalg.norm(C.c + C.g - c) + r
            res1 = tmp < C.r or withinTol(tmp, C.r)
            tmp = np.linalg.norm(C.c - C.g - c) + r
            res2 = tmp < C.r or withinTol(tmp, C.r)
            
            return res1 or res2


def _aux_contains_point(C, p):
    """
    Checks if a point is contained in the capsule
    
    Args:
        C: Capsule object
        p: Point to check
        
    Returns:
        bool: True if point is contained in capsule
    """
    # Get object properties
    g = C.g
    r = C.r
    c = C.c
    
    # Ensure p is a column vector
    p = np.asarray(p).reshape(-1, 1) if np.asarray(p).ndim == 1 else np.asarray(p)
    
    # Check case where capsule is a hyperball
    if g is None or np.allclose(g, 0):
        return _aux_in_sphere(c, r, p)
    else:
        ng = np.linalg.norm(g)
        g_ = g / ng
        proj = np.dot((p - c).T, g_).item()  # Use .item() to get scalar
        
        # Check if point is in hypercylinder
        if abs(proj) < np.linalg.norm(g):
            # Compute distance to axis, check if smaller than the radius
            dist = np.linalg.norm((c + proj * g_) - p)
            return dist < r or withinTol(dist, r)
        else:
            # Check if point is in upper or lower hypersphere
            return _aux_in_sphere(c + g, r, p) or _aux_in_sphere(c - g, r, p)


def _aux_in_sphere(c, r, p):
    """
    Checks if a point is contained in the hypersphere defined by center and radius
    
    Args:
        c: Center of sphere
        r: Radius of sphere
        p: Point to check
        
    Returns:
        bool: True if point is in sphere
    """
    # Ensure both c and p are column vectors
    c = np.asarray(c).reshape(-1, 1) if np.asarray(c).ndim == 1 else np.asarray(c)
    p = np.asarray(p).reshape(-1, 1) if np.asarray(p).ndim == 1 else np.asarray(p)
    
    tmp = np.linalg.norm(p - c)
    return tmp < r or withinTol(tmp, r) 