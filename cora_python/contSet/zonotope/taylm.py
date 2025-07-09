"""
taylm method for zonotope class
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def taylm(Z: Zonotope):
    """
    Enclose a zonotope object with a Taylor model
    
    Args:
        Z: zonotope object
        
    Returns:
        Taylm object
        
    Example:
        Z = Zonotope(np.array([[1], [0]]), np.array([[1, -1], [0, 1]]))
        t = taylm(Z)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Create taylor models for factors
    m = Z.G.shape[1]
    dom = interval(-np.ones((m, 1)), np.ones((m, 1)))
    t = taylm(dom)
    
    # Create taylor model for the zonotope
    t = Z.c + Z.G @ t
    
    return t


def interval(lower: np.ndarray, upper: np.ndarray):
    """
    Create an interval object (placeholder for actual interval implementation)
    """
    # This is a placeholder - in a full implementation, this would create
    # an actual interval object from the cora_python.contSet.interval module
    return {'lower': lower, 'upper': upper}


def taylm(dom):
    """
    Create a Taylor model object (placeholder for actual taylm implementation)
    """
    # This is a placeholder - in a full implementation, this would create
    # an actual taylm object from the cora_python.contSet.taylm module
    return {'domain': dom} 