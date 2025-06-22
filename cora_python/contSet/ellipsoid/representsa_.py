"""
This module contains the function for checking if an ellipsoid can be represented as another set type.
"""

import numpy as np
from typing import Union, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def representsa_(E: 'Ellipsoid', type_str: str, tol: float = 1e-9, *args) -> Union[bool, Tuple[bool, Any]]:
    """
    Checks if an ellipsoid can also be represented by a different set, e.g., a special case.
    
    Args:
        E: ellipsoid object
        type_str: other set representation or 'origin', 'point', 'hyperplane'
        tol: tolerance
        *args: additional arguments (unused)
    
    Returns:
        res: true/false
        S: converted set (if second output requested)
    """
    from cora_python.contSet.emptySet.emptySet import EmptySet
    
    # Check if we need to return the converted set
    return_set = len(args) > 0 and args[0] == 'return_set'
    
    # Check empty object case
    if E.Q is None or E.Q.size == 0:
        if type_str == 'emptySet':
            if return_set:
                return True, EmptySet(E.dim() if hasattr(E, 'n') and E.n is not None else 0)
            return True
        else:
            if return_set:
                return False, None
            return False
    
    # Dimension
    n = E.dim()
    
    # Init second output argument (covering all cases with res = False)
    S = None
    
    # Is the ellipsoid just a point?
    is_point = np.allclose(E.Q, 0, atol=tol)
    
    if type_str == 'origin':
        res = is_point and E.q is not None and np.allclose(E.q, 0, atol=tol)
        if return_set and res:
            S = np.zeros((n, 1))
        
    elif type_str == 'point':
        res = is_point
        if return_set and res:
            S = E.q.copy() if E.q is not None else np.zeros((n, 1))
            
    elif type_str == 'capsule':
        # Only if ellipsoid is 1D, a point, or a ball
        if is_point:
            res = True
        elif n == 1:
            res = True
        else:
            # Check if it's a ball (diagonal matrix with equal entries)
            diag_Q = np.diag(E.Q)
            is_diagonal = np.allclose(E.Q - np.diag(diag_Q), 0, atol=tol)
            all_equal_diag = np.allclose(diag_Q, diag_Q[0], atol=tol)
            res = is_diagonal and all_equal_diag
        
        if return_set and res:
            raise CORAerror('CORA:notSupported', 
                          f'Conversion from ellipsoid to {type_str} not supported.')
    
    elif type_str == 'conHyperplane':
        # Only a constrained hyperplane if ellipsoid is 1D or a point
        res = n == 1 or is_point
        if return_set and res:
            raise CORAerror('CORA:notSupported',
                          f'Conversion from ellipsoid to {type_str} not supported.')
    
    elif type_str == 'conPolyZono':
        raise CORAerror('CORA:notSupported',
                      f'Comparison of ellipsoid to {type_str} not supported.')
    
    elif type_str == 'conZonotope':
        # Only a constrained zonotope if ellipsoid is 1D or a point
        res = n == 1 or is_point
        if return_set and res:
            raise CORAerror('CORA:notSupported',
                          f'Conversion from ellipsoid to {type_str} not supported.')
    
    elif type_str == 'ellipsoid':
        # Obviously true
        res = True
        if return_set:
            S = E
    
    elif type_str == 'halfspace':
        # Ellipsoids cannot be unbounded
        res = False
    
    elif type_str == 'interval':
        # Only an interval if ellipsoid is 1D or a point
        res = n == 1 or is_point
        if return_set and res:
            from cora_python.contSet.ellipsoid.interval import interval
            S = interval(E)
    
    elif type_str == 'levelSet':
        raise CORAerror('CORA:notSupported',
                      f'Comparison of ellipsoid to {type_str} not supported.')
    
    elif type_str == 'polytope':
        # Only a polytope if ellipsoid is 1D or a point
        res = n == 1 or is_point
        if return_set and res:
            raise CORAerror('CORA:notSupported',
                          f'Conversion from ellipsoid to {type_str} not supported.')
    
    elif type_str == 'polyZonotope':
        raise CORAerror('CORA:notSupported',
                      f'Comparison of ellipsoid to {type_str} not supported.')
    
    elif type_str == 'probZonotope':
        res = False
    
    elif type_str == 'zonoBundle':
        # Only a zonotope bundle if ellipsoid is 1D or a point
        res = n == 1 or is_point
        if return_set and res:
            raise CORAerror('CORA:notSupported',
                          f'Conversion from ellipsoid to {type_str} not supported.')
    
    elif type_str == 'zonotope':
        # Only a zonotope if ellipsoid is 1D or a point
        res = n == 1 or is_point
        if return_set and res:
            from cora_python.contSet.ellipsoid.zonotope import zonotope
            S = zonotope(E)
    
    elif type_str == 'hyperplane':
        # Ellipsoid cannot be unbounded (unless 1D, where hyperplane also bounded)
        res = n == 1
    
    elif type_str == 'parallelotope':
        raise CORAerror('CORA:notSupported',
                      f'Comparison of ellipsoid to {type_str} not supported.')
    
    elif type_str == 'convexSet':
        res = True
    
    elif type_str == 'emptySet':
        res = (E.Q is None or E.Q.size == 0) and (E.q is None or E.q.size == 0)
        if return_set and res:
            S = EmptySet(n)
    
    elif type_str == 'fullspace':
        # Ellipsoid cannot be unbounded
        res = False
    
    else:
        raise CORAerror('CORA:wrongValue',
                      f'Unknown set representation: {type_str}')
    
    if return_set:
        return res, S
    return res 