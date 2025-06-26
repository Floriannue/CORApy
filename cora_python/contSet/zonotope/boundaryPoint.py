"""
boundaryPoint method for zonotope class
"""

import numpy as np
from typing import Optional, Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from .zonotope import Zonotope


def boundaryPoint(Z: Zonotope, dir: np.ndarray, startPoint: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Computes the point on the boundary of a zonotope along a given direction,
    starting from a given start point, or, by default, from the center of the set.
    Note that the vector may immediately reach the boundary of degenerate zonotopes.
    
    Args:
        Z: zonotope object
        dir: direction vector
        startPoint: start point for the direction vector (default: center of Z)
        
    Returns:
        Point on the boundary of the zonotope
        
    Raises:
        CORAerror: If direction vector is zero or start point not contained in set
    """
    # Default starting point is the center
    if startPoint is None:
        startPoint = Z.c
    
    # Check if direction vector is non-zero
    if np.allclose(dir, 0):
        raise CORAerror("CORA:wrongValue", 
                       "Vector has to be non-zero.")
    
    # Check for dimensions
    equal_dim_check(Z, dir)
    equal_dim_check(Z, startPoint)
    
    # Read out dimension
    n = Z.dim()
    
    # Empty set
    if Z.representsa_('emptySet', 0):
        return np.zeros((n, 0))
    
    # Start point must be contained in the set
    if not Z.contains_(startPoint, 'exact', 1e-8, 0, False, False)[0]:
        raise CORAerror('CORA:wrongValue',
                       'Start point must be contained in the set.')
    
    # 1D zonotope (start point has no effect)
    if n == 1:
        if dir[0] < 0:
            x = Z.c - np.sum(np.abs(Z.G))
        elif dir[0] > 0:
            x = Z.c + np.sum(np.abs(Z.G))
        else:
            x = Z.c  # dir[0] == 0, but we checked for non-zero above
        return x.flatten()
    
    # Use zonotope norm if start point is the center (default case)
    if np.allclose(Z.c.flatten(), startPoint.flatten(), atol=1e-12):
        # Translate by the center
        Z_atorigin = Zonotope(np.zeros((n, 1)), Z.G)
        
        # Compute boundary point using the zonotope norm
        norm_val = Z_atorigin.zonotopeNorm(dir.reshape(-1, 1))[0]
        x = Z.c.flatten() + dir.flatten() / norm_val
        return x
    
    # If start point is another point contained in the zonotope, we formulate
    # the boundary point computation as a linear program:
    #   max t  s.t.  startPoint + t*dir ∈ Z
    # which can be formulated as
    #   min_{t,beta} -t 
    #   s.t.  s + l*t = c + G beta  <=>  l*t - G beta = c - s
    #         t >= 0, -1 <= beta <= 1
    numGen = Z.G.shape[1]
    
    # Objective function: minimize -t (to maximize t)
    f = np.zeros(1 + numGen)
    f[0] = -1
    
    # Equality constraint: dir*t - G*beta = c - startPoint
    Aeq = np.hstack([dir.reshape(-1, 1), -Z.G])
    beq = Z.c.flatten() - startPoint.flatten()
    
    # Bounds: t >= 0, -1 <= beta <= 1
    lb = np.zeros(1 + numGen)
    lb[1:] = -1
    ub = np.full(1 + numGen, np.inf)
    ub[1:] = 1
    
    # Solve linear program
    problem = {
        'f': f,
        'Aeq': Aeq,
        'beq': beq,
        'Aineq': None,
        'bineq': None,
        'lb': lb,
        'ub': ub
    }
    
    # We should not run into any problems as the start point is guaranteed to
    # be contained within the zonotope (by the check above)
    x_opt, fval, exitflag, output, lambda_out = CORAlinprog(problem)
    
    # Compute solution (note the -1*fval because of optimizing -t)
    if fval is not None:
        t_opt = -fval
    else:
        t_opt = x_opt[0] if x_opt is not None else 0
    
    x = startPoint.flatten() + t_opt * dir.flatten()
    
    return x 