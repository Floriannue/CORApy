import numpy as np
from typing import Union, Tuple, Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.emptySet.emptySet import EmptySet
    from cora_python.contSet.zonotope.zonotope import Zonotope
    from cora_python.contSet.interval.interval import Interval
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def representsa_(capsule_obj, set_type: str, tol: float = 1e-12, **kwargs) -> Union[bool, Tuple[bool, Any]]:
    """
    Checks if a capsule can also be represented by a different set, e.g., a special case
    
    Args:
        capsule_obj: The Capsule object.
        set_type: other set representation or 'origin', 'point', 'hyperplane'
        tol: tolerance
        
    Returns:
        res - true/false
        S - converted set (optional)
    """
    return_set = 'return_set' in kwargs and kwargs['return_set']
    
    # Check empty object case
    if capsule_obj.is_empty():
        if set_type == 'emptySet':
            res = True
        elif set_type in ['interval', 'zonotope', 'ellipsoid', 'polytope', 'conZonotope', 'zonoBundle']:
            # Empty sets can represent these set types
            res = True
        else:
            res = False
        if return_set and res:
            if set_type == 'emptySet':
                from cora_python.contSet.emptySet.emptySet import EmptySet
                S = EmptySet(capsule_obj.dim())
            elif set_type == 'interval':
                from cora_python.contSet.interval.interval import Interval
                S = Interval.empty(capsule_obj.dim())
            elif set_type == 'zonotope':
                from cora_python.contSet.zonotope.zonotope import Zonotope
                S = Zonotope.empty(capsule_obj.dim())
            # Add other empty set conversions as needed
        elif return_set and not res:
            S = None
        if return_set:
            return res, S
        else:
            return res

    # default values
    res = False
    S = None

    if set_type == 'zonotope':
        # MATLAB: res = n == 1 || withinTol(C.r,0,tol);
        # Only a zonotope if 1D or no radius
        res = (capsule_obj.dim() == 1) or (capsule_obj.r < tol)
        if return_set and res:
            from cora_python.contSet.zonotope.zonotope import Zonotope
            if capsule_obj.dim() == 1:
                # 1D case
                S = Zonotope(capsule_obj.c, capsule_obj.g)
            elif np.all(np.abs(capsule_obj.g) < tol):
                # Point case
                S = Zonotope(capsule_obj.c, np.zeros((capsule_obj.dim(), 0)))
            else:
                # Line segment case
                S = Zonotope(capsule_obj.c, capsule_obj.g)

    elif set_type == 'interval':
        # MATLAB: res = n == 1 || (withinTol(C.r,0,tol) && nnz(withinTol(C.g,0,tol)) >= n-1);
        # Only an interval if 1D or (no radius and generator is axis-aligned)
        n = capsule_obj.dim()
        if n == 1:
            res = True
        else:
            # Check if no radius and generator is axis-aligned (at most one non-zero component)
            g_nonzero = np.abs(capsule_obj.g.flatten()) >= tol
            num_nonzero = np.sum(g_nonzero)
            res = (capsule_obj.r < tol) and (num_nonzero <= 1)
        
        if return_set and res:
            from cora_python.contSet.interval.interval import Interval
            if n == 1:
                # 1D case
                if capsule_obj.r < tol:
                    # Line segment
                    min_val = capsule_obj.c - np.abs(capsule_obj.g)
                    max_val = capsule_obj.c + np.abs(capsule_obj.g)
                else:
                    # Ball in 1D
                    min_val = capsule_obj.c - capsule_obj.r
                    max_val = capsule_obj.c + capsule_obj.r
                S = Interval(min_val, max_val)
            else:
                # Axis-aligned line segment
                min_val = capsule_obj.c - np.abs(capsule_obj.g)
                max_val = capsule_obj.c + np.abs(capsule_obj.g)
                S = Interval(min_val, max_val)

    elif set_type == 'ellipsoid':
        # Capsule is an ellipsoid if g is zero (it's a ball)
        if np.all(np.abs(capsule_obj.g) < tol):
            res = True
            if return_set:
                from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
                # An ellipsoid is defined by center and shape matrix P
                # For a ball, P is r^2 * I
                P = (capsule_obj.r**2) * np.eye(capsule_obj.dim())
                S = Ellipsoid(P, capsule_obj.c)

    elif set_type == 'origin':
        # Check if capsule represents the origin
        # A capsule represents origin if it's a point AND the center is at origin
        is_point = (np.all(np.abs(capsule_obj.g) < tol) and capsule_obj.r < tol)
        res = is_point and np.all(np.abs(capsule_obj.c) < tol)

    elif set_type == 'point':
        # A capsule is a point if g is zero and r is zero
        res = (np.all(np.abs(capsule_obj.g) < tol) and capsule_obj.r < tol)
        if return_set and res:
            S = capsule_obj.c

    elif set_type == 'hyperplane':
        # A capsule is a hyperplane if its dimension is 1 and it has infinite extent (g or r is infinite, which is not supported here)
        # For finite capsules, they can't be hyperplanes.
        res = False

    if return_set:
        return res, S
    else:
        return res 