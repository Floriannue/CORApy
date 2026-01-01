"""
brusselator - brusselator system from Example 3.4.1 in [1]

Syntax:
    f = brusselator(x, u)

Inputs:
    x - state vector
    u - input vector

Outputs:
    f - time-derivative of the system state

Reference:
   [1] X. Chen. "Reachability Analysis of Non-Linear Hybrid Systems Using
       Taylor Models"

Authors:       Niklas Kochdumper
Written:       19-June-2020
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np


def brusselator(x, u):
    """
    brusselator - brusselator system dynamics
    
    Args:
        x: state vector (2-dimensional)
        u: input vector
        
    Returns:
        f: time-derivative of the system state
    """
    f = np.zeros((2, 1))
    f[0, 0] = 1 + x[0, 0]**2 * x[1, 0] - 2.5 * x[0, 0]
    f[1, 0] = 1.5 * x[0, 0] - x[0, 0]**2 * x[1, 0]
    
    return f

