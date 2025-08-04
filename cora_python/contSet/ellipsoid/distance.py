"""
distance - computes the distance between an ellipsoid and the another set
    representation or a point

Syntax:
    val = distance(E,S)

Inputs:
    E - ellipsoid object
    S - contSet object, numeric, cell-array

Outputs:
    val - distance(s) between ellipsoid and set/point

Example:
    E = ellipsoid(eye(2))
    P = polytope([1 1]/sqrt(2),-2)
    distance(E,P)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       08-March-2021 (MATLAB)
Last update:   18-March-2021 (allowing cell arrays, MATLAB)
               04-July-2022 (VG, replace cell arrays by class arrays, MATLAB)
               05-October-2024 (MW, remove class arrays, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, List, Tuple
from cora_python.contSet.contSet import ContSet
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from .private.priv_distancePoint import priv_distancePoint
from .private.priv_distancePolytope import priv_distancePolytope
from .private.priv_distanceHyperplane import priv_distanceHyperplane
from .private.priv_distanceEllipsoid import priv_distanceEllipsoid
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric as reorderNumeric
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check as equalDimCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def distance(factor1, factor2):
    """
    Computes the distance between an ellipsoid and another set representation or a point
    
    Args:
        factor1: ellipsoid object or numeric
        factor2: contSet object, numeric, or cell array
        
    Returns:
        val: distance(s) between ellipsoid and set/point
    """
    E, S = reorderNumeric(factor1, factor2)
    
    # Check input arguments - use proper class validation
    # Since we know E should be an Ellipsoid, let's check it directly
    if not isinstance(E, Ellipsoid):
        raise CORAerror('CORA:wrongValue', 'first', 'Expected ellipsoid object')
    
    # For S, we allow multiple types, so check more flexibly
    if not (isinstance(S, (ContSet, np.ndarray, list)) or hasattr(S, '__iter__')):
        raise CORAerror('CORA:wrongValue', 'second', 'Expected contSet object, numeric array, or list')
    
    # Check equal dimensions
    equalDimCheck(E, S)
    
    if isinstance(S, np.ndarray):
        return priv_distancePoint(E, S)
    
    # Rewrite S as cell-array for easier handling
    if not isinstance(S, list):
        S = [S]
    
    # Loop over individual pairs
    val = np.zeros(len(S))
    for i in range(len(S)):
        current_S = S[i]
        if isinstance(current_S, np.ndarray):
            result = priv_distancePoint(E, current_S)
            val[i] = result.item() if hasattr(result, 'item') else result
        else:
            result = aux_distance(E, current_S)
            val[i] = result.item() if hasattr(result, 'item') else result
    
    # Return scalar if single element
    if len(val) == 1:
        return val[0]
    return val

def aux_distance(E, S):
    """
    Auxiliary function to compute distance between ellipsoid and single set
    
    Args:
        E: ellipsoid object
        S: single contSet object
        
    Returns:
        val: distance value
    """
    if S.representsa_('emptySet', E.TOL):
        # Distance to empty set = 0 since empty-set \subseteq obj
        return 0.0
    
    # Different distances
    if isinstance(S, Ellipsoid):
        return priv_distanceEllipsoid(E, S)
    
    # Import here to avoid circular imports
    try:
        from cora_python.contSet.polytope import Polytope
        if isinstance(S, Polytope):
            if S.representsa_('hyperplane', E.TOL):
                return priv_distanceHyperplane(E, S)
            else:
                return priv_distancePolytope(E, S)
    except ImportError:
        pass
    
    # Check for polytope-like object with different name
    if hasattr(S, '__class__') and 'polytope' in S.__class__.__name__.lower():
        if hasattr(S, 'representsa_') and S.representsa_('hyperplane', E.TOL):
            return priv_distanceHyperplane(E, S)
        else:
            return priv_distancePolytope(E, S)
    
    # For other unsupported types, return NotImplementedError for now
    # In MATLAB this would throw CORAerror('CORA:noops',E,S)
    raise TypeError(f"Operation 'distance' not supported between instances of '{type(E).__name__}' and '{type(S).__name__}'") 