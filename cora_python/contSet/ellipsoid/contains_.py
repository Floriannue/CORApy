"""
This module contains the function for checking if an ellipsoid contains a set or points.
"""

import numpy as np
from typing import Union, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .private.priv_containsPoint import priv_containsPoint
from .private.priv_containsEllipsoid import priv_containsEllipsoid


def contains_(E: 'Ellipsoid', S: Union[np.ndarray, Any], method: str = 'exact', 
              tol: float = 1e-9, max_eval: int = 1000, cert_toggle: bool = False, 
              scaling_toggle: bool = False, *args) -> Union[bool, Tuple[bool, bool, float]]:
    """
    Determines if an ellipsoid contains a set or a point.
    
    Args:
        E: ellipsoid object
        S: contSet object or single point (numpy array)
        method: method used for the containment check ('exact' or 'approx')
        tol: tolerance for the containment check
        max_eval: maximum evaluations (currently has no effect)
        cert_toggle: if True, cert will be computed, otherwise set to NaN
        scaling_toggle: if True, scaling will be computed, otherwise set to inf
        *args: additional arguments
    
    Returns:
        res: true/false
        cert: returns true iff the result could be verified (if requested)
        scaling: smallest scaling factor (if requested)
    """
    from cora_python.contSet.ellipsoid.representsa_ import representsa_
    
    # Validate method parameter
    if method not in ['exact', 'approx']:
        raise CORAerror('CORA:wrongValue', f"method must be 'exact' or 'approx', got '{method}'")
    
    cert = True
    scaling = 0.0
    
    # Check trivial cases
    # If E is a point...
    try:
        ell_is_point, p = representsa_(E, 'point', tol, 'return_set')
    except:
        ell_is_point = False
        p = None
        
    if ell_is_point:
        if isinstance(S, np.ndarray):
            # S is numeric
            res = np.max(np.abs(S - p)) <= tol
            cert = True
            if res:
                scaling = 0.0
            else:
                scaling = np.inf
        else:
            # S is not numeric, check if S is a point (but as a contSet)
            try:
                S_is_point, q = representsa_(S, 'point', tol, 'return_set')
                if S_is_point:
                    res = np.allclose(q, p, atol=tol)
                    cert = True
                    if res:
                        scaling = 0.0
                    else:
                        scaling = np.inf
                elif representsa_(S, 'emptySet', tol):
                    # If S is not a point at all, test that it is not the empty set
                    res = True
                    cert = True
                    scaling = 0.0
                else:
                    # S is not numeric and not empty -> S cannot possibly be contained
                    res = False
                    cert = True
                    scaling = np.inf
            except:
                res = False
                cert = True
                scaling = np.inf
        
        if cert_toggle or scaling_toggle:
            return res, cert, scaling
        return res
    
    # E is not a point
    # Check if E is empty
    if representsa_(E, 'emptySet', tol):
        if isinstance(S, np.ndarray):
            # If S is numeric, check manually whether it is empty
            if S.size == 0:
                res = True
                scaling = 0.0
                cert = True
            else:
                # If it is not empty, it cannot possibly be contained
                res = False
                scaling = np.inf
                cert = True
        else:
            # S is a set - check if it represents an empty set (polymorphic)
            try:
                if hasattr(S, 'representsa_') and callable(S.representsa_):
                    is_empty = S.representsa_('emptySet', tol)
                elif hasattr(S, 'is_empty') and callable(S.is_empty):
                    is_empty = S.is_empty()
                else:
                    is_empty = False
            except:
                is_empty = False
                
            if is_empty:
                res = True
                scaling = 0.0
                cert = True
            else:
                # If it is not empty, it cannot possibly be contained
                res = False
                scaling = np.inf
                cert = True
        
        if cert_toggle or scaling_toggle:
            return res, cert, scaling
        return res
    
    # E is not empty
    if isinstance(S, np.ndarray):
        res, cert, scaling = priv_containsPoint(E, S, tol)
        if cert_toggle or scaling_toggle:
            return res, cert, scaling
        return res
    
    # Check if S is unbounded
    if hasattr(S, 'isBounded') and callable(S.isBounded):
        try:
            if not S.isBounded():
                # Unbounded -> not contained, since E is always bounded
                res = False
                cert = True
                scaling = np.inf
                if cert_toggle or scaling_toggle:
                    return res, cert, scaling
                return res
        except:
            pass
    
    # Check if S represents an empty set (polymorphic)
    try:
        if hasattr(S, 'representsa_') and callable(S.representsa_):
            is_empty = S.representsa_('emptySet', tol)
        elif hasattr(S, 'is_empty') and callable(S.is_empty):
            is_empty = S.is_empty()
        else:
            is_empty = False
    except:
        is_empty = False
    
    if is_empty:
        # Empty -> always contained
        res = True
        cert = True
        scaling = 0.0
        if cert_toggle or scaling_toggle:
            return res, cert, scaling
        return res
    else:
        try:
            # Check if S represents a point (polymorphic)
            if hasattr(S, 'representsa_') and callable(S.representsa_):
                is_point, p = S.representsa_('point', tol, 'return_set')
                if is_point:
                    res, cert, scaling = priv_containsPoint(E, p, tol)
                    if cert_toggle or scaling_toggle:
                        return res, cert, scaling
                    return res
        except Exception as e:
            if 'CORA:notSupported' in str(e) or 'MATLAB:maxlhs' in str(e):
                # If the code above returns an error either because there are
                # too many outputs, or the operation is not supported, we
                # conclude that it is not implemented, and we do nothing
                pass
            else:
                # In any other case, something went wrong. Relay that information.
                raise e
    
    # Containment check for specific set types
    if hasattr(S, '__class__'):
        class_name = S.__class__.__name__
        
        if class_name == 'Ellipsoid':
            res, cert, scaling = priv_containsEllipsoid(E, S, tol)
            if cert_toggle or scaling_toggle:
                return res, cert, scaling
            return res
        
        elif class_name == 'Capsule':
            # Check if balls at both ends of capsule are contained
            try:
                from cora_python.contSet.ellipsoid.dim import dim
                n = dim(S)
                
                # Create ellipsoids for both ends
                E1 = type(E)((S.r ** 2) * np.eye(n), S.c + S.g)
                E2 = type(E)((S.r ** 2) * np.eye(n), S.c - S.g)
                
                res1, cert1, scaling1 = priv_containsEllipsoid(E, E1, tol)
                res2, cert2, scaling2 = priv_containsEllipsoid(E, E2, tol)
                
                res = res1 and res2
                cert = True
                scaling = max(scaling1, scaling2)
                
                if cert_toggle or scaling_toggle:
                    return res, cert, scaling
                return res
            except:
                res = False
                cert = False
                scaling = np.inf
                if cert_toggle or scaling_toggle:
                    return res, cert, scaling
                return res
    
    # Handle other set types based on method
    if method == 'exact':
        # Check for sets that support vertex enumeration
        vertex_supported_types = ['conZonotope', 'interval', 'polytope', 'zonoBundle']
        if hasattr(S, '__class__') and S.__class__.__name__ in vertex_supported_types:
            # Check if all vertices of the set are contained
            try:
                vertices_S = S.vertices()
                res, cert, scaling = priv_containsPoint(E, vertices_S, tol)
                res = np.all(res) if isinstance(res, np.ndarray) else res
                cert = True
                scaling = np.max(scaling) if isinstance(scaling, np.ndarray) else scaling
                
                if cert_toggle or scaling_toggle:
                    return res, cert, scaling
                return res
            except:
                pass
        
        elif hasattr(S, '__class__') and S.__class__.__name__ == 'zonotope':
            # For zonotopes, we can leverage the symmetry for better vertex enumeration
            try:
                from cora_python.contSet.ellipsoid.dim import dim
                if dim(S) <= 2:
                    vertices_S = S.vertices()
                    res, cert, scaling = priv_containsPoint(E, vertices_S, tol)
                    res = np.all(res) if isinstance(res, np.ndarray) else res
                    cert = True
                    scaling = np.max(scaling) if isinstance(scaling, np.ndarray) else scaling
                else:
                    # Use more sophisticated algorithm for higher dimensions
                    res, cert, scaling = _priv_venum_zonotope(E, S, tol, scaling_toggle)
                
                if cert_toggle or scaling_toggle:
                    return res, cert, scaling
                return res
            except:
                pass
        
        # If we reach here, throw error for unsupported exact algorithm
        raise CORAerror('CORA:noExactAlg', f'No exact algorithm available for ellipsoid contains {type(S)}')
    
    elif method == 'approx':
        # Compute approx algorithms
        if hasattr(S, '__class__') and S.__class__.__name__ == 'zonotope':
            res, cert, scaling = _aux_symmetric_grothendieck(E, S, tol, cert_toggle)
            if cert_toggle or scaling_toggle:
                return res, cert, scaling
            return res
        elif hasattr(S, 'zonotope') and callable(S.zonotope):
            # Convert to zonotope and try again
            res, cert, scaling = contains_(E, S.zonotope(), method, tol, max_eval, cert_toggle, scaling_toggle)
            cert = res  # For approximation, cert equals res
            if cert_toggle or scaling_toggle:
                return res, cert, scaling
            return res
        else:
            raise CORAerror('CORA:noExactAlg', f'No approximation algorithm available for ellipsoid contains {type(S)}')
    else:
        raise CORAerror('CORA:noSpecificAlg', f'No algorithm available for ellipsoid contains {type(S)} with method {method}')


def _priv_venum_zonotope(E: 'Ellipsoid', Z, tol: float, scaling_toggle: bool) -> Tuple[bool, bool, float]:
    """
    Vertex enumeration for zonotope containment (simplified implementation).
    """
    # This is a placeholder for the complex vertex enumeration algorithm
    # Full implementation would be quite involved
    try:
        vertices_Z = Z.vertices()
        res, cert, scaling = priv_containsPoint(E, vertices_Z, tol)
        res = np.all(res) if isinstance(res, np.ndarray) else res
        cert = True
        scaling = np.max(scaling) if isinstance(scaling, np.ndarray) else scaling
        return res, cert, scaling
    except:
        return False, False, np.inf


def _aux_symmetric_grothendieck(E: 'Ellipsoid', Z, tol: float, cert_toggle: bool) -> Tuple[bool, bool, float]:
    """
    Symmetric Grothendieck approach for zonotope containment (simplified implementation).
    """
    # This is a placeholder for the complex SDP-based algorithm
    # Full implementation would require semidefinite programming solvers
    try:
        # Conservative fallback to vertex-based approach
        vertices_Z = Z.vertices()
        res, cert, scaling = priv_containsPoint(E, vertices_Z, tol)
        res = np.all(res) if isinstance(res, np.ndarray) else res
        cert = res  # For approximation methods
        scaling = np.max(scaling) if isinstance(scaling, np.ndarray) else scaling
        return res, cert, scaling
    except:
        return False, False, np.inf 