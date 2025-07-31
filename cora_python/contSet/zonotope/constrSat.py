"""
constrSat - checks if all points x within a zonotope satisfy a linear inequality constraint

Syntax:
    res = constrSat(Z,C,d)

Inputs:
    Z - zonotope object
    C - normal vectors of constraints
    d - distance to origin of constraints

Outputs:
    res - boolean whether constraint is satisfied

Example:
    Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
    C = np.array([[1, 1]])
    d = np.array([2])
    res = constrSat(Z, C, d)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff
Written:      10-August-2011
Last update:  14-May-2017
Last revision:---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Optional
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.interval import interval


def constrSat(Z: Zonotope, C: np.ndarray, d: np.ndarray) -> bool:
    """
    Checks if all points x within a zonotope satisfy a linear inequality constraint
    
    Args:
        Z: zonotope object
        C: normal vectors of constraints
        d: distance to origin of constraints
        
    Returns:
        Boolean whether constraint is satisfied
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Validate input dimensions
    if C.shape[1] != Z.c.shape[0]:
        raise CORAerror('CORA:wrongDimensions', 
                       'Dimension of constraint matrix C does not match zonotope dimension')
    
    if d.shape[0] != C.shape[0]:
        raise CORAerror('CORA:wrongDimensions', 
                       'Dimension of constraint vector d does not match number of constraints')
    
    # Compute C*Z first (following MATLAB order)
    CZ = C @ Z
    
    # Convert to interval
    CZ_interval = interval(CZ)
    I = Interval(CZ_interval.inf - d, CZ_interval.sup - d)
    
    if I is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Could not compute constraint interval')
    
    supremum_val = I.supremum()
    if supremum_val is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Could not compute interval supremum')
    
    # Check if interval contains 0
    res = bool(np.all(supremum_val < 0))
    
    return res 