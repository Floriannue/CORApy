"""
taylm - enclose a zonotope object with a Taylor model

Syntax:
    t = taylm(Z)

Inputs:
    Z - zonotope object

Outputs:
    t - taylm object

Example:
    from cora_python.contSet.zonotope import Zonotope, taylm
    import numpy as np
    Z = Zonotope(np.array([[1], [0]]), np.array([[1, -1], [0, 1]]))
    t = taylm(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       13-August-2018 (MATLAB)
Last update:   --- (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""
import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def taylm(Z: Zonotope):
    """
    Enclose a zonotope object with a Taylor model.
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Create taylor models for factors
    m = Z.G.shape[1]
    from cora_python.contSet.interval.interval import Interval
    from cora_python.contSet.taylm.taylm import Taylm
    
    dom = Interval(-np.ones((m, 1)), np.ones((m, 1)))
    t = Taylm(dom)
    
    # Create taylor model for the zonotope
    t = Z.c + Z.G @ t
    
    return t 