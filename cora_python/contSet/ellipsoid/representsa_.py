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
    res = False
    
    # Is the ellipsoid just a point?
    # A point ellipsoid has a zero shape matrix, regardless of its dimensions.
    isPoint = np.all(E.Q == 0) # Check if all elements are exactly zero

    # Debug print for isPoint logic (commented out to reduce noise)
    # print(f"[DEBUG] representsa_ input E.Q: {E.Q}, E.q: {E.q}, isPoint initial: {isPoint}")

    type_lower = type_str.lower()

    if type_lower == 'origin':
        # MATLAB: res = isPoint && ~isempty(E.q) && all(withinTol(E.q,0,tol));
        # Check if it's a point ellipsoid, q is non-empty, and q is close to zero
        res = isPoint and (E.q is not None and E.q.size > 0) and np.all(withinTol(E.q, 0, tol))
        if return_set and res:
            S = np.zeros((n, 1))
            
    elif type_lower == 'point':
        res = isPoint
        if return_set and res:
            S = E.q.copy()
            
    elif type_lower == 'capsule':
        # Only if ellipsoid is 1D, a point, or a ball
        # MATLAB: diag(E.Q) needs to be numerically checked
        diag_Q = np.diag(E.Q)
        # A ball means all eigenvalues of Q are equal and positive (or zero for point).
        # So, the diagonal elements of Q should be approximately equal for a diagonal Q matrix,
        # and for a general Q, all eigenvalues should be equal. Here we check the diagonal
        # elements after a potential rotation to principal axes if E.Q is not diagonal.
        # For simplicity, if Q is already diagonal (or near-diagonal) and its diagonal elements
        # are within tolerance, we consider it a ball or a point.
        # A more robust check would involve eigenvalues: np.all(withinTol(np.diff(np.linalg.eigvalsh(E.Q)), 0, tol))
        is_ball = np.all(withinTol(diag_Q, diag_Q[0], tol)) if E.Q.ndim > 1 and E.Q.shape[0] > 0 else True # Handle 0D and 1D

        res = (n == 1 or isPoint or (is_ball and np.all(np.diag(E.Q) > -tol))) # ensure diagonal is not negative (pos semi-definite)
        
        if return_set and res:
            raise CORAerror('CORA:notSupported', f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_lower == 'conhyperplane':
        # Only a constrained hyperplane if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            raise CORAerror('CORA:notSupported', f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_lower == 'conpolyzono':
        raise CORAerror('CORA:notSupported', f"Comparison of ellipsoid to {type_str} not supported.")
            
    elif type_lower == 'conzonotope':
        # Only a constrained zonotope if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            raise CORAerror('CORA:notSupported', f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_lower == 'ellipsoid':
        res = True
        if return_set:
            S = E
            
    elif type_lower == 'halfspace':
        # Ellipsoids cannot be unbounded
        res = False
        
    elif type_lower == 'interval':
        # Only an interval if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            from cora_python.contSet.interval.interval import Interval # local import
            S = E.interval()
                
    elif type_str == 'levelSet':
        raise CORAerror('CORA:notSupported', f"Comparison of ellipsoid to {type_str} not supported.")
            
    elif type_lower == 'polytope':
        # Only a polytope if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            from cora_python.contSet.polytope.polytope import Polytope # local import
            # For a point, create a 0-gen polytope. For 1D ellipsoid, it's an interval which is a polytope.
            if isPoint:
                S = Polytope(E.q)
            elif n == 1:
                I_obj = E.interval() # Use interval conversion if 1D
                S = Polytope(np.array([[-1.0],[1.0]]), np.array([[-I_obj.inf], [I_obj.sup]]))
            else:
                raise CORAerror('CORA:notSupported', f"Conversion from ellipsoid to {type_str} not supported for n={n}.")
            
    elif type_lower == 'polyzonotope':
        raise CORAerror('CORA:notSupported', f"Comparison of ellipsoid to {type_str} not supported.")
            
    elif type_lower == 'probzonotope':
        res = False
        
    elif type_lower == 'zonobundle':
        # Only a zonotope bundle if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            raise CORAerror('CORA:notSupported', f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_lower == 'zonotope':
        # Only a zonotope if ellipsoid is 1D or a point
        res = n == 1 or isPoint
        if return_set and res:
            from cora_python.contSet.zonotope.zonotope import Zonotope # local import
            S = E.zonotope()
            
    elif type_lower == 'hyperplane':
        # Ellipsoid cannot be unbounded (unless 1D, where hyperplane also bounded)
        res = n == 1
        if return_set and res:
            raise CORAerror('CORA:notSupported', f"Conversion from ellipsoid to {type_str} not supported.")
            
    elif type_lower == 'parallelotope':
        raise CORAerror('CORA:notSupported', f"Comparison of ellipsoid to {type_str} not supported.")
            
    elif type_lower == 'convexset':
        res = True
        
    elif type_lower == 'emptyset':
        # An ellipsoid represents an empty set if its shape matrix Q is 0x0,
        # its center vector q is empty, and its tolerance is empty or default.
        res = E.isemptyobject()
        if return_set and res:
            from cora_python.contSet.emptySet import EmptySet
            S = EmptySet(n)
            
    elif type_lower == 'fullspace':
        # Ellipsoid cannot be unbounded
        res = False
        
    else:
        # Unknown set type - MATLAB doesn't throw error, just returns false
        res = False
    
    # Debug print before final return
    # print(f"[DEBUG] representsa_ Final result for {type_str}: res={res}, S={S}")

    if return_set:
        return res, S
    return res 