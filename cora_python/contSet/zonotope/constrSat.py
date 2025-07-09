"""
constrSat method for zonotope class
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def constrSat(Z: Zonotope, C: np.ndarray, d: np.ndarray) -> bool:
    """
    Checks if all points x within a zonotope satisfy a linear inequality constraint
    
    Args:
        Z: zonotope object
        C: normal vectors of constraints
        d: distance to origin of constraints
        
    Returns:
        Boolean whether constraint is satisfied
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        C = np.array([[1, 1]])
        d = np.array([2])
        res = constrSat(Z, C, d)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Check if constraints are violated
    from cora_python.contSet.interval.interval import interval
    
    # Compute C*Z
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Zonotope center or generators are None')
    
    # Compute interval of C*Z
    Z_interval = interval(Z)
    if Z_interval is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Could not create interval from zonotope')
    
    # Compute C*Z_interval
    C_Z = C @ Z_interval
    
    # Add -d to get I = interval(C*Z) + (-d)
    I = C_Z + (-d)
    
    # Check if interval contains 0
    if I is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Could not compute constraint interval')
    
    supremum_val = I.supremum()
    if supremum_val is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Could not compute interval supremum')
    
    res = np.all(supremum_val < 0)
    
    return res 