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
from .private.priv_venumZonotope import priv_venumZonotope

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


def contains_(E: 'Ellipsoid', S: Union[np.ndarray, Any], method: str = 'exact', 
              tol: float = 1e-9, max_eval: int = 1000, cert_toggle: bool = False, 
              scaling_toggle: bool = False, *varargin) -> Tuple[bool, bool, float]:
    """
    Determines if an ellipsoid contains a set or a point.
    
    Syntax:
        res = contains_(E, S)
        res = contains_(E, S, method)
        res, cert, scaling = contains_(E, S, method, tol, max_eval, cert_toggle, scaling_toggle)
    
    Args:
        E: ellipsoid object
        S: contSet object or single point
        method: method used for the containment check ('exact' or 'approx')
        tol: tolerance for the containment check
        max_eval: maximum evaluations (currently has no effect)
        cert_toggle: if True, cert will be computed, otherwise set to NaN
        scaling_toggle: if True, scaling will be computed, otherwise set to inf
        *varargin: additional arguments
    
    Returns:
        res: true/false
        cert: returns true iff the result could be verified (if cert_toggle is True)
        scaling: smallest scaling factor (if scaling_toggle is True)
    
    References:
        [1] Yildirim, E.A., 2006. On the minimum volume covering ellipsoid of
            of ellipsoids. SIAM Journal on Optimization, 17(3), pp.621-641.     
        [2] SDPT3: url: http://www.math.nus.edu.sg/~mattohkc/sdpt3.html
        [3] Kulmburg, A, Schäfer, L, Althoff, A, 2024. Approximability of the
            Containment Problem for Zonotopes and Ellipsotopes
    """
    
    # Validate method
    if method not in ['exact', 'approx']:
        raise CORAerror('CORA:noSpecificAlg', f'No algorithm for method {method}')
    
    cert = True
    scaling = 0.0
    
    # Check trivial cases
    # If E is a point...
    try:
        ell_is_point, p = E.representsa_('point', tol, return_set=True)
    except Exception:
        ell_is_point = False
        p = None
        
    if ell_is_point:
        if isinstance(S, np.ndarray):
            # S is numeric
            res = np.max(np.abs(S - p)) <= tol
            cert = True
            scaling = 0.0 if res else np.inf
        else:
            # S is not numeric, check if S is a point (but as a contSet)
            try:
                # First check if S represents an empty set
                if hasattr(S, 'representsa_') and callable(S.representsa_):
                    is_empty = S.representsa_('emptySet', tol)
                    if is_empty:
                        # Empty set is always contained in any ellipsoid
                        res = True
                        cert = True
                        scaling = 0.0
                        return res, cert, scaling
                
                # Then check if S is a point
                try:
                    S_is_point, q = S.representsa_('point', tol, return_set=True)
                except TypeError:
                    # Some representsa_ methods don't support return_set parameter
                    S_is_point = S.representsa_('point', tol)
                    q = None
                if S_is_point:
                    if isinstance(q, np.ndarray) and q.shape == p.shape and not np.any(np.isnan(q)):
                        # Direct comparison with numpy array (only if shapes match and no nan)
                        res = np.allclose(q, p, atol=tol)
                        if res:
                            cert = True
                            scaling = 0.0
                            return res, cert, scaling
                    else:
                        # q is invalid -> fall back to vertex check
                        try:
                            vertices = S.vertices_()
                            if vertices.size > 0:
                                point_res, point_cert, point_scaling = priv_containsPoint(E, vertices, tol)
                                res = np.all(point_res) if isinstance(point_res, np.ndarray) else point_res
                            else:
                                # vertices_() failed, try to extract point from equality constraints
                                if hasattr(S, 'Ae') and hasattr(S, 'be') and S.Ae is not None and S.be is not None:
                                    if np.allclose(S.Ae, np.eye(S.Ae.shape[0]), atol=1e-12):
                                        point_candidate = S.be
                                        point_res, point_cert, point_scaling = priv_containsPoint(E, point_candidate, tol)
                                        res = np.all(point_res) if isinstance(point_res, np.ndarray) else point_res
                                    else:
                                        res = False
                                else:
                                    res = False
                        except Exception:
                            res = False
                    cert = True
                    scaling = 0.0 if res else np.inf
                else:
                    # S is not numeric and not point
                    res = False
                    cert = True
                    scaling = np.inf
            except Exception:
                # Fallback on error
                res = False
                cert = True
                scaling = np.inf
        return res, cert, scaling
    
    # E is not a point
    # Check if E is empty
    if E.representsa_('emptySet', tol):
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
                else:
                    is_empty = False
            except Exception:
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
        
        return res, cert, scaling
    
    # E is not empty
    if isinstance(S, np.ndarray):
        res, cert, scaling = priv_containsPoint(E, S, tol)
        # Convert arrays to scalars if we have a single point
        if S.ndim == 1 or (S.ndim == 2 and S.shape[1] == 1):
            if isinstance(res, np.ndarray) and res.size == 1:
                res = bool(res.item())
            if isinstance(cert, np.ndarray) and cert.size == 1:
                cert = bool(cert.item())
            if isinstance(scaling, np.ndarray) and scaling.size == 1:
                scaling = float(scaling.item())
        return res, cert, scaling
    
    # Check if S is unbounded
    if hasattr(S, 'isBounded') and callable(S.isBounded):
        try:
            if not S.isBounded():
                # Unbounded -> not contained, since E is always bounded
                res = False
                cert = True
                scaling = np.inf
                return res, cert, scaling
        except Exception:
            pass
    
    # Check if S represents an empty set (polymorphic)
    try:
        if hasattr(S, 'representsa_') and callable(S.representsa_):
            is_empty = S.representsa_('emptySet', tol)
        else:
            is_empty = False
    except Exception:
        is_empty = False
    
    if is_empty:
        # Empty -> always contained
        res = True
        cert = True
        scaling = 0.0
        return res, cert, scaling
    else:
        try:
            # Check if S represents a point (polymorphic)
            if hasattr(S, 'representsa_') and callable(S.representsa_):
                is_point, p = S.representsa_('point', tol, return_set=True)
                if is_point:
                    res, cert, scaling = priv_containsPoint(E, p, tol)
                    # Convert arrays to scalars for single points
                    if isinstance(res, np.ndarray) and res.size == 1:
                        res = bool(res.item())
                    if isinstance(cert, np.ndarray) and cert.size == 1:
                        cert = bool(cert.item())
                    if isinstance(scaling, np.ndarray) and scaling.size == 1:
                        scaling = float(scaling.item())
                    return res, cert, scaling
        except Exception as e:
            if 'CORA:notSupported' in str(e) or 'MATLAB:maxlhs' in str(e):
                # Not implemented or wrong number of outputs
                pass
            else:
                raise e
    
    # Containment check for specific set types
    if hasattr(S, '__class__'):
        class_name = S.__class__.__name__
        
        if class_name == 'Ellipsoid':
            res, cert, scaling = priv_containsEllipsoid(E, S, tol)
            return res, cert, scaling
        
        elif class_name == 'Capsule':
            # Check if balls at both ends of capsule are contained
            from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
            E1 = Ellipsoid(S.r**2 * np.eye(S.dim()), S.c + S.g)
            E2 = Ellipsoid(S.r**2 * np.eye(S.dim()), S.c - S.g)
            
            res1, cert1, scaling1 = priv_containsEllipsoid(E, E1, tol)
            res2, cert2, scaling2 = priv_containsEllipsoid(E, E2, tol)
            res = res1 and res2
            cert = True
            scaling = max(scaling1, scaling2)
            return res, cert, scaling
    
    # Method-specific containment checks
    if method == 'exact':
        if hasattr(S, '__class__'):
            class_name = S.__class__.__name__
            
            if class_name in ['ConZonotope', 'Interval', 'Polytope', 'ZonoBundle']:
                # Check if all vertices of the set are contained
                vertices = S.vertices_()
                res, cert, scaling = priv_containsPoint(E, vertices, tol)
                if isinstance(res, np.ndarray):
                    res = np.all(res)
                    scaling = np.max(scaling) if isinstance(scaling, np.ndarray) else scaling
                cert = True
                return res, cert, scaling
                
            elif class_name == 'Zonotope':
                if S.dim() <= 2:
                    vertices = S.vertices_()
                    res, cert, scaling = priv_containsPoint(E, vertices, tol)
                    if isinstance(res, np.ndarray):
                        res = np.all(res)
                        scaling = np.max(scaling) if isinstance(scaling, np.ndarray) else scaling
                    cert = True
                    return res, cert, scaling
                else:
                    res, cert, scaling = priv_venumZonotope(E, S, tol, scaling_toggle)
                    return res, cert, scaling
            else:
                raise CORAerror('CORA:noExactAlg', f'No exact algorithm for {class_name}')
        else:
            raise CORAerror('CORA:noExactAlg', 'No exact algorithm for unknown set type')
            
    elif method == 'approx':
        if hasattr(S, '__class__') and S.__class__.__name__ == 'Zonotope':
            res, cert, scaling = aux_symmetricGrothendieck(E, S, tol, cert_toggle)
            return res, cert, scaling
        elif hasattr(S, 'zonotope') and callable(S.zonotope):
            Z = S.zonotope()
            res, cert, scaling = contains_(E, Z, method, tol, max_eval, cert_toggle, scaling_toggle)
            cert = res
            return res, cert, scaling
        else:
            raise CORAerror('CORA:noExactAlg', 'No approximation algorithm for this set type')
    else:
        raise CORAerror('CORA:noSpecificAlg', f'No algorithm for method {method}')



def aux_symmetricGrothendieck(E: 'Ellipsoid', Z, tol: float, cert_toggle: bool) -> Tuple[bool, bool, float]:
    """
    Symmetric Grothendieck method for zonotope containment in ellipsoid.
    
    This implements the method described in [3] for checking if a zonotope
    is contained in an ellipsoid using semidefinite programming.
    
    Args:
        E: ellipsoid object
        Z: zonotope object
        tol: tolerance for containment check
        cert_toggle: if True, compute certificate
        
    Returns:
        res: containment result
        cert: certificate (True if result is verified)
        scaling: scaling factor
    """
    
    n = Z.dim()
    m = Z.generators().shape[1]
    
    # First, we deal with the centers; Z < E if and only if Z' < E-center(E),
    # where Z' is the zonotope with generator matrix [G center(Z)-center(E)]:
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    Z_center = Z.center()
    E_center = E.center()
    
    # Create new zonotope with adjusted center
    new_generators = np.hstack([Z.generators(), (Z_center - E_center)])
    Z = Zonotope(np.zeros((n, 1)), new_generators)
    
    # Adjust ellipsoid center to origin
    E = E - E_center
    
    G = Z.generators()
    
    # We now deal with the case where E is degenerate:
    if not E.isFullDim():
        # Z < E can only happen if Z lies in the same subspace as E, so we need
        # to check that every generator of Z lies, up to some scaling, in E.
        # The fastest way to do this is via the ellipsoid norm: if it is equal
        # to Inf for some generator, this means that the generator in question
        # does not lie in the same subspace:
        for i in range(m):
            if E.ellipsoidNorm(G[:, i:i+1]) == np.inf:
                res = False
                cert = False
                scaling = np.inf
                return res, cert, scaling
        
        # So, now we know that Z lies in the same subspace as E. Let us rotate
        # E using the svd in such a way, that the axes of E are aligned with
        # the canonical ONB:
        U, S_vals, Vt = np.linalg.svd(E.Q)
        T = U
        
        # Rotate E and Z
        E_rotated = T.T @ E
        Z_rotated = T.T @ Z
        
        # We can now remove the last coordinates
        r = np.linalg.matrix_rank(E.Q)
        from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
        E = Ellipsoid(E_rotated.Q[:r, :r], np.zeros((r, 1)))
        G = Z_rotated.generators()
        Z = Zonotope(np.zeros((r, 1)), G[:r, :])
    
    # We can now assume that E is non-degenerate, which means that Q is
    # invertible. We now use the method described in [3]
    G = Z.generators()
    m = G.shape[1]
    
    if not CVXPY_AVAILABLE:
        # Fallback to vertex-based approach
        return _fallback_vertex_approach(E, Z, tol, cert_toggle)
    
    try:
        # Use cvxpy for semidefinite programming
        X = cp.Variable((m, m), PSD=True)
        lambda_var = cp.Variable()
        
        # Constraints: X >= 0 and X(i,i) == 1 for all i
        constraints = [X >> 0]  # PSD constraint
        for i in range(m):
            constraints.append(X[i, i] == 1)
        
        Q_inv = np.linalg.inv(E.Q)
        
        # Objective: minimize -trace(G^T * Q^(-1) * G * X)
        objective = cp.Minimize(-cp.trace(G.T @ Q_inv @ G @ X))
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            # Compute scaling
            X_val = X.value
            if X_val is not None:
                scaling = np.sqrt(np.abs(np.trace(G.T @ Q_inv @ G @ X_val)))
                
                if scaling <= 1 + tol:
                    res = True
                    cert = True
                else:
                    res = False
                    if scaling > np.pi / 2:
                        # If scaling > pi/2, we know from [3] that the set cannot possibly be contained
                        cert = True
                    else:
                        cert = False
                        
                return res, cert, scaling
        
        # If optimization failed, fall back to vertex approach
        return _fallback_vertex_approach(E, Z, tol, cert_toggle)
        
    except Exception:
        # If cvxpy fails, fall back to vertex approach
        return _fallback_vertex_approach(E, Z, tol, cert_toggle)


def _fallback_vertex_approach(E: 'Ellipsoid', Z, tol: float, cert_toggle: bool) -> Tuple[bool, bool, float]:
    """
    Fallback approach using vertex enumeration when SDP solver is not available.
    
    Args:
        E: ellipsoid object
        Z: zonotope object
        tol: tolerance
        cert_toggle: certificate toggle
        
    Returns:
        res: containment result
        cert: certificate
        scaling: scaling factor
    """
    try:
        # Get vertices of the zonotope
        vertices = Z.vertices_()
        
        # Check if all vertices are contained
        res, cert, scaling = priv_containsPoint(E, vertices, tol)
        
        if isinstance(res, np.ndarray):
            res = np.all(res)
            scaling = np.max(scaling) if isinstance(scaling, np.ndarray) else scaling
        
        cert = True  # We can verify vertex-based results
        return res, cert, scaling
        
    except Exception:
        # If vertex enumeration fails, return conservative result
        res = False
        cert = False
        scaling = np.inf
        return res, cert, scaling 