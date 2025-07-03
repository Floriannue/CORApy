"""
representsa_ - check if an interval represents a specific set type

Syntax:
    res = representsa_(I, type, tol)
    res, S = representsa_(I, type, tol)

Inputs:
    I - interval object
    type - string specifying the set type
    tol - tolerance for comparison

Outputs:
    res - true if interval represents the specified type, false otherwise
    S - converted set (if second output requested)

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 19-July-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Union, Any, Optional
from .interval import Interval
from cora_python.g.functions.matlab.validate.check import withinTol 

def representsa_(obj: Interval, set_type: str, tol: float = 1e-9, **kwargs) -> Union[bool, Tuple[bool, Any]]:
    """
    Check if interval represents a specific set type
    
    Args:
        obj: Interval object
        set_type: Type of set to check ('emptySet', 'origin', 'point', 'fullspace', etc.)
        tol: Tolerance for comparison
        **kwargs: Additional arguments
        
    Returns:
        bool or tuple: True if interval represents the specified type, 
                      or (bool, converted_set) if return_set=True
    """
    return_set = kwargs.get('return_set', False)
    
    # Normalize set_type to handle case-insensitive comparison
    set_type = set_type.lower()
    
    # Check empty object case using base class method
    empty, res, S = obj.representsa_emptyObject(set_type)
    if empty:
        if return_set:
            return res, S
        return res
    
    # Get dimension
    n = obj.dim()
    
    # Initialize second output argument (covering all cases with res = false)
    S = None
    
    # Exclude matrix interval cases
    shape = obj.inf.shape
    if len(shape) == 0:
        r, c = 1, 1  # Scalar case
    elif len(shape) == 1:
        r, c = shape[0], 1
    elif len(shape) == 2:
        r, c = shape
    else:
        # N-dimensional array - treat as matrix for exclusion purposes
        # Use first two dimensions as r, c
        r, c = shape[0], shape[1]
    # Allow fullspace check for matrix intervals since they can represent fullspace
    if (set_type not in ['emptyset', 'origin', 'fullspace'] and r > 1 and c > 1):
        raise ValueError(
            "representsa only supports vector interval objects (except type = 'emptySet', 'origin', 'fullspace')."
        )
    
    # Check if interval is a point
    radius = obj.rad()
    is_point = np.all(withinTol(radius, 0, tol))
    
    # Handle different set types using switch-like structure
    if set_type == 'origin':
        res = (obj.inf.size > 0 and 
               np.all(withinTol(obj.inf, 0, tol)) and 
               np.all(withinTol(obj.sup, 0, tol)))
        if return_set and res:
            S = np.zeros((r, c))
    
    elif set_type == 'point':
        res = is_point
        if return_set and res:
            S = obj.center()
    
    elif set_type == 'capsule':
        # Either 1D, a point, or at most one dimension has non-zero width
        # This means at least (n-1) dimensions should have zero width
        res = (n == 1 or is_point or 
               np.sum(withinTol(radius, 0, tol)) >= n - 1)
        if return_set and res:
            # Would need capsule implementation
            from cora_python.contSet.capsule import Capsule
            S = Capsule(obj)
    
    elif set_type == 'conhyperplane':
        # Only if 1D, a point, or at least one dimension has zero width
        # MATLAB: nnz(withinTol(rad(I),0,tol)) >= 1
        # This counts the number of zero-width dimensions
        res = (n == 1 or is_point or 
               np.sum(withinTol(radius, 0, tol)) >= 1)
        # No conversion supported
    
    elif set_type == 'conpolyzono':
        res = True
        if return_set:
            from cora_python.contSet.zonotope.zonotope import Zonotope
            from cora_python.contSet.conPolyZono import ConPolyZono
            # Convert interval to zonotope first
            c = obj.center()
            G = np.diag(obj.rad().flatten())
            Z = Zonotope(c, G)
            S = ConPolyZono(Z)
    
    elif set_type == 'conzonotope':
        res = True
        if return_set:
            from cora_python.contSet.zonotope.zonotope import Zonotope
            from cora_python.contSet.conZonotope import ConZonotope
            # Convert interval to zonotope first
            c = obj.center()
            G = np.diag(obj.rad().flatten())
            Z = Zonotope(c, G)
            S = ConZonotope(Z)
    
    elif set_type == 'ellipsoid':
        # Either 1D, a point, or at most one dimension has non-zero width
        # This means at least (n-1) dimensions should have zero width
        res = (n == 1 or is_point or 
               np.sum(withinTol(radius, 0, tol)) >= n - 1)
        if return_set and res:
            raise NotImplementedError("Conversion from interval to ellipsoid not supported")
    
    elif set_type == 'halfspace':
        # Only if interval is bounded in exactly one direction
        inf_flat = obj.inf.flatten()
        sup_flat = obj.sup.flatten()
        total_infinite = np.sum(np.isinf(inf_flat)) + np.sum(np.isinf(sup_flat))
        res = total_infinite == 2 * n - 1
        if return_set and res:
            raise NotImplementedError("Conversion from interval to halfspace not supported")
    
    elif set_type == 'interval':
        # Obviously true
        res = True
        if return_set:
            S = obj
    
    elif set_type == 'levelset':
        res = True
        if return_set:
            # No direct transformation from interval to levelSet available
            from cora_python.contSet.polytope import Polytope
            from cora_python.contSet.levelSet import LevelSet
            S = LevelSet(Polytope(obj))
    
    elif set_type == 'polytope':
        res = True
        if return_set:
            from cora_python.contSet.polytope import Polytope
            S = Polytope(obj)
    
    elif set_type == 'polyzonotope':
        res = True
        if return_set:
            from cora_python.contSet.zonotope.zonotope import Zonotope
            from cora_python.contSet.polyZonotope import PolyZonotope
            # Convert interval to zonotope first
            c = obj.center()
            G = np.diag(obj.rad().flatten())
            Z = Zonotope(c, G)
            S = PolyZonotope(Z)
    
    elif set_type == 'probzonotope':
        # Cannot be true
        res = False
    
    elif set_type == 'zonobundle':
        res = True
        if return_set:
            from cora_python.contSet.zonoBundle import ZonoBundle
            S = ZonoBundle(obj)
    
    elif set_type == 'zonotope':
        res = True
        if return_set:
            from cora_python.contSet.zonotope.zonotope import Zonotope
            c = obj.center()
            G = np.diag(obj.rad().flatten())
            S = Zonotope(c, G)
    
    elif set_type == 'hyperplane':
        # Exactly one dimension has to be zero-width
        res = np.sum(withinTol(radius, 0, tol)) == 1
        if return_set and res:
            raise NotImplementedError("Conversion from interval to hyperplane not supported")
    
    elif set_type == 'parallelotope':
        res = True
        if return_set:
            from cora_python.contSet.zonotope.zonotope import Zonotope
            c = obj.center()
            G = np.diag(obj.rad().flatten())
            S = Zonotope(c, G)
    
    elif set_type == 'convexset':
        res = True
    
    elif set_type == 'emptyset':
        # Already handled in isemptyobject
        res = False
    
    elif set_type == 'fullspace':
        res = (np.all(obj.inf == -np.inf) and np.all(obj.sup == np.inf))
        if return_set and res:
            from cora_python.contSet.fullspace import Fullspace
            S = Fullspace(n)
    
    else:
        # Unknown set type
        res = False
    
    if return_set:
        return res, S
    return res

