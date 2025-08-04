"""
constrSat - checks if all points x within a zonotope satisfy a linear
    inequality constraint

Syntax:
    res = constrSat(Z, C, d)

Inputs:
    Z - zonotope object
    C - normal vectors of constraints
    d - distance to origin of constraints

Outputs:
    res - boolean whether constraint is satisfied

Example:

Other m-files required:
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       10-August-2011 (MATLAB)
Last update:   14-May-2017 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Optional
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.interval.interval import Interval


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

    # Compute C*Z first (following MATLAB order)
    CZ = C @ Z
    
    # Convert to interval
    CZ_interval = CZ.interval()
    
    # Add (-d) to the interval: I = interval(C*Z) + (-d)
    I = CZ_interval + (-d)
    
    # Check if interval contains 0: res = all(supremum(I) < 0)
    res = bool(np.all(I.supremum() < 0))
    
    return res 