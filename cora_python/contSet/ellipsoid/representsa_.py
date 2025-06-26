"""
This module contains the function for checking if an ellipsoid can be represented as another set type.
"""

import numpy as np
import inspect
import ast
from typing import Union, Tuple, Any, TYPE_CHECKING
from scipy.linalg import eigvals

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

def representsa_(E: 'Ellipsoid', type_str: str, tol: float = 1e-9, **kwargs):
    """
    Checks if an ellipsoid can represent a certain set type.
    
    Args:
        E: Ellipsoid object
        type_str: String representing the set type ('emptySet', 'point', etc.)
        tol: Tolerance for numerical comparisons
        **kwargs: Additional arguments
        
    Returns:
        bool or tuple: True/False or (True/False, converted_set) if return_set=True
    """
    return_set = kwargs.get('return_set', False)
    
    # Check empty object case using base class method
    empty, res, S = E.representsa_emptyObject(type_str)
    if empty:
        if return_set:
            return res, S
        return res
    
    # Dimension
    n = E.dim()
    
    # Initialize second output argument
    S = None
    
    # Is the ellipsoid just a point?
    isPoint = np.all(np.all(withinTol(E.Q, 0, tol)))
    
    if type_str == 'origin':
        res = isPoint and E.q is not None and np.all(withinTol(E.q, 0, tol))
        if return_set and res:
            S = np.zeros((n, 1))
            
    elif type_str == 'point':
        res = isPoint
        if return_set and res:
            S = E.q.copy()
            
    elif type_str == 'capsule':
        # Only if ellipsoid is 1D, a point, or a ball
        diagEQ = np.diag(E.Q)
        res = (n == 1 or isPoint or 
               (np.count_nonzero(E.Q) == n and np.all(withinTol(diagEQ, diagEQ[0], tol))))
        if return_set and res:
            raise NotImplementedError(f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_str == 'conHyperplane':
        # Only a constrained hyperplane if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            raise NotImplementedError(f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_str == 'conPolyZono':
        raise NotImplementedError(f"Comparison of ellipsoid to {type_str} not supported.")
            
    elif type_str == 'conZonotope':
        # Only a constrained zonotope if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            raise NotImplementedError(f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_str == 'ellipsoid':
        res = True
        if return_set:
            S = E
            
    elif type_str == 'halfspace':
        # Ellipsoids cannot be unbounded
        res = False
        
    elif type_str == 'interval':
        # Only an interval if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            from cora_python.contSet.ellipsoid.interval import interval
            S = interval(E)
                
    elif type_str == 'levelSet':
        raise NotImplementedError(f"Comparison of ellipsoid to {type_str} not supported.")
            
    elif type_str == 'polytope':
        # Only a polytope if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            raise NotImplementedError(f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_str == 'polyZonotope':
        raise NotImplementedError(f"Comparison of ellipsoid to {type_str} not supported.")
            
    elif type_str == 'probZonotope':
        res = False
        
    elif type_str == 'zonoBundle':
        # Only a zonotope bundle if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            raise NotImplementedError(f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_str == 'zonotope':
        # Only a zonotope if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            from cora_python.contSet.ellipsoid.zonotope import zonotope
            S = zonotope(E)
            
    elif type_str == 'hyperplane':
        # Ellipsoid cannot be unbounded (unless 1D, where hyperplane also bounded)
        res = n == 1
        if return_set and res:
            raise NotImplementedError(f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_str == 'parallelotope':
        raise NotImplementedError(f"Comparison of ellipsoid to {type_str} not supported.")
            
    elif type_str == 'convexSet':
        res = True
        
    elif type_str == 'emptySet':
        res = (E.Q is None or E.Q.size == 0) and (E.q is None or E.q.size == 0)
        if return_set and res:
            from cora_python.contSet.emptySet import EmptySet
            S = EmptySet(n)
            
    elif type_str == 'fullspace':
        # Ellipsoid cannot be unbounded
        res = False
        
    else:
        # Unknown set type - MATLAB doesn't throw error, just returns false
        res = False
    
    if return_set:
        return res, S
    return res 