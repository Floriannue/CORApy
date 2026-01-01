"""
fiveDimSysEq - linear system from [1, Sec. 3.2.3] modelled as a nonlinear
               system (required for unit test)

Syntax:
    dx = fiveDimSysEq(x, u)

Inputs:
    x - state vector
    u - input vector

Outputs:
    dx - time-derivative of the system state

References:
    [1] M. Althoff, "Reachability analysis and its application to the 
        safety assessment of autonomous cars", Dissertation, TUM 2010

Authors:       Niklas Kochdumper
Written:       19-June-2020
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np


def fiveDimSysEq(x, u):
    """
    fiveDimSysEq - linear system modelled as a nonlinear system
    
    Args:
        x: state vector (5-dimensional)
        u: input vector (5-dimensional)
        
    Returns:
        dx: time-derivative of the system state
    """
    dx = np.zeros((5, 1))
    dx[0, 0] = -x[0, 0] - 4 * x[1, 0] + u[0, 0]
    dx[1, 0] = 4 * x[0, 0] - x[1, 0] + u[1, 0]
    dx[2, 0] = -3 * x[2, 0] + x[3, 0] + u[2, 0]
    dx[3, 0] = -x[2, 0] - 3 * x[3, 0] + u[3, 0]
    dx[4, 0] = -2 * x[4, 0] + u[4, 0]
    
    return dx

