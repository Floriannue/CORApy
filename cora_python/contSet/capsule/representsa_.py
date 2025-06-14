import numpy as np
from typing import Union, Tuple, Any

from cora_python.contSet.emptySet import empty as emptySet
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


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
        else:
            res = False
        if return_set and res:
            S = emptySet(capsule_obj.dim())
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
        if np.all(np.abs(capsule_obj.g) < tol):
            # Case: Capsule is a ball (g is zero)
            res = True
            if return_set:
                S = Zonotope(capsule_obj.c, np.zeros((capsule_obj.dim(), 0)), capsule_obj.r)
        elif capsule_obj.r < tol:
            # Case: Capsule is a line segment (r is zero)
            res = True
            if return_set:
                S = Zonotope(capsule_obj.c, capsule_obj.g)
        else:
            # general case: Capsule is a zonotope
            # C = c + [-g, g] + Ball(r)
            # this is a zonotope of order 2, g_1 = g, g_2 = r * eye(dim)
            res = True
            if return_set:
                # This is a bit of a simplification, a capsule can be represented
                # as a zonotope, but it's an over-approximation of a ball.
                # The actual representation of a ball as a zonotope needs care.
                # For now, we approximate by adding generators representing the ball.
                G_ball = np.eye(capsule_obj.dim()) * capsule_obj.r
                G = np.hstack((capsule_obj.g, G_ball))
                S = Zonotope(capsule_obj.c, G)

    elif set_type == 'interval':
        # Capsule is an interval if g is axis aligned and r is 0
        if capsule_obj.r < tol and np.sum(np.abs(capsule_obj.g) > tol) == 1:
            res = True
            if return_set:
                # min/max for interval
                min_val = capsule_obj.c - np.abs(capsule_obj.g)
                max_val = capsule_obj.c + np.abs(capsule_obj.g)
                S = Interval(min_val, max_val)

    elif set_type == 'ellipsoid':
        # Capsule is an ellipsoid if g is zero (it's a ball)
        if np.all(np.abs(capsule_obj.g) < tol):
            res = True
            if return_set:
                # An ellipsoid is defined by center and shape matrix P
                # For a ball, P is r^2 * I
                P = (capsule_obj.r**2) * np.eye(capsule_obj.dim())
                S = Ellipsoid(capsule_obj.c, P)

    elif set_type == 'origin':
        # Capsule contains origin if 0 is in the set
        # For capsule, this means ||c||_2 <= r + ||g||_1 (this is a rough over-approximation)
        # A more precise check for 0 in capsule:
        # Distance from origin to line segment L is d(0, L)
        # If d(0, L) <= r, then origin is in capsule
        
        # If the origin is within the interval [-|g|, |g|] + c, and also within the ball of radius r, then it contains the origin.
        # This is a basic check. More rigorous methods involve projection.
        
        # Check if origin is contained within the ball for a ball, or segment for segment
        if np.all(np.abs(capsule_obj.g) < tol):
            # It's a ball
            if np.linalg.norm(capsule_obj.c) <= capsule_obj.r + tol:
                res = True
        elif capsule_obj.r < tol:
            # It's a line segment
            # Check if origin is on the line segment defined by c and g
            # Project origin onto the line defined by c and g. Check if projection is on segment.
            # Then check if projection distance is 0.
            t_proj = -np.dot(capsule_obj.g.T, capsule_obj.c) / (np.linalg.norm(capsule_obj.g)**2 + 1e-18)
            if -1 <= t_proj <= 1:
                point_on_segment = capsule_obj.c + t_proj * capsule_obj.g
                if np.linalg.norm(point_on_segment) < tol:
                    res = True
        else:
            # General capsule case: Check if origin is contained
            # This is complex and might require optimization/projection
            # For now, a conservative check: if the center is close to origin and r is large enough to cover g
            if np.linalg.norm(capsule_obj.c) <= capsule_obj.r + np.sum(np.abs(capsule_obj.g)) + tol:
                 res = True

    elif set_type == 'point':
        # A capsule is a point if g is zero and r is zero
        if np.all(np.abs(capsule_obj.g) < tol) and capsule_obj.r < tol:
            res = True
            if return_set:
                S = capsule_obj.c

    elif set_type == 'hyperplane':
        # A capsule is a hyperplane if its dimension is 1 and it has infinite extent (g or r is infinite, which is not supported here)
        # For finite capsules, they can't be hyperplanes.
        res = False

    if return_set:
        return res, S
    else:
        return res 