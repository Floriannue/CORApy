"""
interval - Over-approximates an ellipsoid by an interval

Syntax:
    I = interval(E)

Inputs:
    E - ellipsoid object

Outputs:
    I - interval object

Example: 
    E = ellipsoid([3 -1; -1 1],[1;0])
    I = interval(E)

    figure hold on
    plot(E)
    plot(I,[1,2],'r')

Other m-files required: interval (zonotope)
Subfunctions: none
MAT-files required: none

See also: vertices, polytope

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       13-March-2019 (MATLAB)
Last update:   04-July-2022 (VG, input checks, MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.init.unitvector import unitvector


def interval(E):
    """
    Over-approximates an ellipsoid by an interval
    
    Args:
        E: ellipsoid object
        
    Returns:
        I: interval object
    """
    # Check inputs
    inputArgsCheck([[E, 'att', 'ellipsoid', 'scalar']])
    
    n = E.dim()
    E0 = Ellipsoid(E.Q, np.zeros_like(E.q))
    
    # Compute width of the ellipsoid in each dimension using support functions
    dI = np.zeros((n, 1))
    for i in range(n):
        val, _ = E0.supportFunc_(unitvector(i + 1, n), 'upper') # unitvector is 1-indexed
        dI[i, 0] = val
    
    # Import here to avoid circular imports
    from cora_python.contSet.interval.interval import Interval
    
    # Construct the resulting interval
    I = Interval(-dI, dI) + E.q
    return I 