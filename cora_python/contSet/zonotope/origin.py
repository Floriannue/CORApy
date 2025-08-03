"""
origin - instantiates a zonotope that contains only the origin

Syntax:
    Z = zonotope.origin(n)

Inputs:
    n - dimension (integer, >= 1)

Outputs:
    Z - zonotope representing the origin

Example: 
    Z = origin(2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       21-September-2024 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from .zonotope import Zonotope

def origin(n):
    """
    Instantiates a zonotope that contains only the origin
    
    Args:
        n: dimension (integer, >= 1)
        
    Returns:
        zonotope: zonotope representing the origin
    """
    
    # Input validation (matching MATLAB)
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'positive', 'integer']]])
    
    return Zonotope(np.zeros((n, 1)))