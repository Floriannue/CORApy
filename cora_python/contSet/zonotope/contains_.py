"""
contains_ - determines if zonotope contains a point/set

Syntax:
    res = contains_(Z, S)
    res = contains_(Z, S, method)
    res = contains_(Z, S, method, tol)
    res, cert = contains_(Z, S, ...)
    res, cert, scaling = contains_(Z, S, ...)

Inputs:
    Z - zonotope object
    S - contSet object or point
    method - (optional) algorithm for the computation:
        'exact' (default), 'approx', 'sampling', 'opt'
    tol - (optional) tolerance
    maxEval - (optional) maximum number of evaluations for sampling
    certToggle - (optional) flag for certification
    scalingToggle - (optional) flag for scaling computation

Outputs:
    res - true/false whether S is contained in Z
    cert - true/false whether result is certain
    scaling - scaling factor

Authors:       Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2006 (MATLAB)
Last update:   many updates
Last revision: 12-March-2021 (MW, add set case)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Union, Tuple, Optional
import warnings


from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

from .compact_ import compact_

from .representsa_ import representsa_


def contains_(Z, S, method: str = 'exact', tol: float = 1e-12, maxEval: int = 200, 
              certToggle: bool = True, scalingToggle: bool = True, *varargin) -> Union[bool, Tuple]:
    """
    Determines if zonotope contains a point or set.
    
    Args:
        Z: zonotope object
        S: point (numpy array) or set object to check
        method: algorithm for computation
        tol: tolerance for containment check
        maxEval: maximum evaluations for sampling methods
        certToggle: whether to return certification
        scalingToggle: whether to compute scaling factor
        
    Returns:
        bool or tuple: containment result, optionally with cert and scaling
    """
    
    # Check if S is a point or a set
    if isinstance(S, np.ndarray):
        # Point containment
        return _point_containment(Z, S, method, tol, certToggle, scalingToggle)
    else:
        # Set containment
        return _set_containment(Z, S, method, tol, maxEval, certToggle, scalingToggle, *varargin)


def _point_containment(Z, p: np.ndarray, method: str, tol: float, 
                      certToggle: bool, scalingToggle: bool) -> Tuple:
    """
    Check if zonotope contains given point(s).
    """
    
    # Import the private function
    from .private.priv_zonotopeContainment_pointContainment import priv_zonotopeContainment_pointContainment
    
    # Ensure p is 2D
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    
    # Check if zonotope is just a point
    if representsa_(Z, 'point', tol):
        # Zonotope is a single point
        if p.shape[1] == 1:
            # Single point query
            res = np.max(np.abs(p - Z.c)) <= tol
            cert = True
            scaling = 0.0 if res else np.inf
        else:
            # Multiple points
            res = np.array([np.max(np.abs(p[:, i:i+1] - Z.c)) <= tol for i in range(p.shape[1])])
            cert = np.ones(p.shape[1], dtype=bool)
            scaling = np.where(res, 0.0, np.inf)
    else:
        # Use private function for general case
        res, cert, scaling = priv_zonotopeContainment_pointContainment(Z, p, method, tol, scalingToggle)
    
    # Always return all three values for consistency
    return res, cert, scaling


def _set_containment(Z, S, method: str, tol: float, maxEval: int, 
                    certToggle: bool, scalingToggle: bool, *varargin) -> Union[bool, Tuple]:
    """
    Check if zonotope contains given set.
    """
    
    # Get the class name of S
    S_class = S.__class__.__name__.lower()
    
    if S_class == 'zonotope':
        return _zonotope_containment(Z, S, method, tol, certToggle, scalingToggle)
    elif S_class == 'interval':
        return _interval_containment(Z, S, method, tol, certToggle, scalingToggle)
    elif S_class == 'polytope':
        return _polytope_containment(Z, S, method, tol, certToggle, scalingToggle)
    else:
        # Try to convert to zonotope first
        try:
            from ..zonotope import Zonotope
            S_zono = Zonotope(S)
            return _zonotope_containment(Z, S_zono, method, tol, certToggle, scalingToggle)
        except:
            raise ValueError(f"Containment check not implemented for {S_class}")


def _zonotope_containment(Z, S_zono, method, tol, certToggle, scalingToggle):
    """Helper function for zonotope-in-zonotope containment"""
    
    if method in ['exact', 'exact:polymax', 'exact:venum']:
        # Fallback to vertex enumeration for exact methods
        vertices = S_zono.vertices()
        # It's a point cloud now, so call the point containment helper
        res, cert, scaling = _point_containment(Z, vertices, 'exact', tol, certToggle, scalingToggle)

        # For set containment, all vertices must be contained.
        # Reduce the array results to a single boolean.
        res_final = np.all(res)
        cert_final = np.all(cert)
        # The required scaling is the maximum scaling over all vertices.
        scaling_final = np.max(scaling) if isinstance(scaling, np.ndarray) and scaling.size > 0 else scaling

        return res_final, cert_final, scaling_final

    elif method == 'approx:st':
        from .private.priv_zonotopeContainment_SadraddiniTedrake import priv_zonotopeContainment_SadraddiniTedrake
        return priv_zonotopeContainment_SadraddiniTedrake(S_zono, Z, tol, scalingToggle)

    else:
        # For other approximation methods, we can try to overapproximate S_zono
        # and check if that overapproximation is contained.
        # This part is complex and requires more of the library to be ported.
        # For now, we'll raise an error for unsupported methods.
        raise ValueError(f"Unsupported method for zonotope-in-zonotope containment: {method}")


def _interval_containment(Z, I, method: str, tol: float, 
                         certToggle: bool, scalingToggle: bool) -> Union[bool, Tuple]:
    """
    Check if zonotope contains interval.
    """
    
    # Convert interval to zonotope and check containment
    from ..zonotope import Zonotope
    I_zono = Zonotope(I)
    return _zonotope_containment(Z, I_zono, method, tol, certToggle, scalingToggle)


def _polytope_containment(Z, P, method: str, tol: float, 
                         certToggle: bool, scalingToggle: bool) -> Union[bool, Tuple]:
    """
    Check if zonotope contains polytope.
    """
    
    # Check vertices of polytope
    vertices = P.vertices_()
    return _point_containment(Z, vertices, method, tol, certToggle, scalingToggle)


def _point_containment_venum(Z, p, tol, scalingToggle):
    """
    Point containment using zonotope norm (linear programming approach).
    """
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    
    num_points = p.shape[1]
    res = np.zeros(num_points, dtype=bool)
    cert = np.ones(num_points, dtype=bool)
    scaling = np.full(num_points, np.inf)
    
    # Get zonotope properties
    try:
        center = Z.center() if hasattr(Z, 'center') else np.zeros((p.shape[0], 1))
        generators = Z.generators() if hasattr(Z, 'generators') else np.array([]).reshape(center.shape[0], 0)
    except:
        # Fallback for basic zonotope structure
        center = getattr(Z, 'c', np.zeros((p.shape[0], 1)))
        generators = getattr(Z, 'G', np.array([]).reshape(center.shape[0], 0))
    
    if generators.size == 0:
        # Zonotope is just a point
        for i in range(num_points):
            point_diff = np.linalg.norm(p[:, i:i+1] - center, np.inf)
            res[i] = point_diff <= tol
            scaling[i] = point_diff if point_diff > 0 else 0
        return res, cert, scaling
    
    # Check for each point whether zonotope norm is <= 1
    for i in range(num_points):
        point = p[:, i:i+1]
        zono_norm = _zonotope_containment_norm(generators, point - center)
        
        scaling[i] = zono_norm
        res[i] = zono_norm <= 1 + tol
        cert[i] = True
    
    if num_points == 1:
        return res[0], cert[0], scaling[0]
    else:
        return res, cert, scaling


def _point_containment_polymax(Z, p, tol, scalingToggle):
    """
    Point containment using polytope conversion (halfspace approach).
    """
    # For now, implement a simplified version using bounding box
    # This is not exact but provides a basic implementation
    
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    
    num_points = p.shape[1]
    res = np.zeros(num_points, dtype=bool)
    cert = np.ones(num_points, dtype=bool)
    scaling = np.full(num_points, np.inf)
    
    # Get zonotope bounding box as approximation
    try:
        center = Z.center() if hasattr(Z, 'center') else np.zeros((p.shape[0], 1))
        generators = Z.generators() if hasattr(Z, 'generators') else np.array([]).reshape(center.shape[0], 0)
    except:
        # Fallback for basic zonotope structure
        center = getattr(Z, 'c', np.zeros((p.shape[0], 1)))
        generators = getattr(Z, 'G', np.array([]).reshape(center.shape[0], 0))
    
    if generators.size == 0:
        # Zonotope is just a point
        for i in range(num_points):
            point_diff = np.linalg.norm(p[:, i:i+1] - center, np.inf)
            res[i] = point_diff <= tol
            scaling[i] = point_diff if point_diff > 0 else 0
        return res, cert, scaling
    
    # Compute bounding box
    gen_sum = np.sum(np.abs(generators), axis=1, keepdims=True)
    lower_bound = center - gen_sum
    upper_bound = center + gen_sum
    
    for i in range(num_points):
        point = p[:, i:i+1]
        
        # Check if point is within bounding box
        within_bounds = np.all(point >= lower_bound - tol) and np.all(point <= upper_bound + tol)
        res[i] = within_bounds
        
        if scalingToggle and within_bounds:
            # Compute approximate scaling
            diff_from_center = point - center
            if np.any(gen_sum > 0):
                scaling[i] = np.max(np.abs(diff_from_center) / (gen_sum + 1e-12))
            else:
                scaling[i] = 0
        elif scalingToggle:
            scaling[i] = np.inf
    
    if num_points == 1:
        return res[0], cert[0], scaling[0]
    else:
        return res, cert, scaling


def _zonotope_containment_norm(generators, p):
    """
    Compute the zonotope containment norm of point p with respect to generators.
    
    This solves the linear program:
    minimize t
    subject to: generators * x = p
                -t <= x_i <= t for all i
    
    The point is contained if the optimal t <= 1.
    
    Args:
        generators: generator matrix (n x m)
        p: point (n x 1)
        
    Returns:
        float: zonotope containment norm (inf if infeasible)
    """
    if generators.size == 0:
        if np.allclose(p, 0):
            return 0
        else:
            return np.inf
    
    n, m = generators.shape
    p = p.flatten()
    
    # Set up linear program
    # Variables: [t, x1, x2, ..., xm]
    c = np.zeros(1 + m)
    c[0] = 1  # minimize t
    
    # Equality constraints: generators * x = p
    A_eq = np.hstack([np.zeros((n, 1)), generators])
    b_eq = p
    
    # Inequality constraints: -t <= x_i <= t for all i
    # This becomes: x_i - t <= 0 and -x_i - t <= 0
    A_ub = np.zeros((2 * m, 1 + m))
    b_ub = np.zeros(2 * m)
    
    for i in range(m):
        # x_i - t <= 0
        A_ub[2*i, 0] = -1  # -t
        A_ub[2*i, 1+i] = 1  # x_i
        
        # -x_i - t <= 0
        A_ub[2*i+1, 0] = -1  # -t
        A_ub[2*i+1, 1+i] = -1  # -x_i
    
    # Bounds: t >= 0, x_i unbounded
    bounds = [(0, None)] + [(None, None)] * m
    
    try:
        # Solve linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            return result.x[0]  # Return optimal t value
        else:
            return np.inf
    except Exception:
        return np.inf


def _exact_parser(Z, S, method, tol, maxEval, certToggle, scalingToggle):
    """
    Choose exact algorithm based on set type.
    """
    # For now, implement basic fallback
    if hasattr(S, '__class__'):
        class_name = S.__class__.__name__.lower()
        
        if class_name in ['interval', 'zonotope']:
            # Convert to zonotope if needed
            if class_name == 'interval':
                # Convert interval to zonotope (simplified)
                S_zono = S  # Assume conversion exists
            else:
                S_zono = S
            
            # Use vertex enumeration for small zonotopes
            if method == 'exact' and hasattr(S, 'dim') and S.dim() >= 4:
                method = 'exact:venum'
            else:
                method = 'exact:polymax'
        else:
            # Choose polymax for other types
            if method == 'exact':
                method = 'exact:polymax'
    
    # For now, return a basic implementation
    # This would need to be expanded with proper algorithms
    warnings.warn("Exact zonotope-in-zonotope containment not fully implemented. Using approximation.")
    return False, False, np.inf


def _approx_parser(Z, S, method, tol, maxEval, certToggle, scalingToggle):
    """
    Choose approximative algorithm based on set type.
    """
    # For now, return a basic implementation
    warnings.warn("Approximative zonotope-in-zonotope containment not fully implemented.")
    return False, False, np.inf


def _sampling_parser(Z, S, method, tol, maxEval, certToggle, scalingToggle):
    """
    Choose sampling algorithm based on set type.
    """
    # For now, return a basic implementation
    warnings.warn("Sampling-based zonotope containment not implemented.")
    return False, False, np.inf


def _opt_parser(Z, S, method, tol, maxEval, certToggle, scalingToggle):
    """
    Optimization-based containment checking.
    """
    # For now, return a basic implementation
    warnings.warn("Optimization-based zonotope containment not implemented.")
    return False, False, np.inf 