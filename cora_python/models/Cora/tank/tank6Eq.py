"""
tank6Eq - system dynamics for the tank benchmark (see Sec. VII in [1])

Syntax:
    dx = tank6Eq(x, u)

Inputs:
    x - state vector
    u - input vector

Outputs:
    dx - time-derivative of the system state

References:
    [1] M. Althoff et al. "Reachability analysis of nonlinear systems with 
        uncertain parameters using conservative linearization", CDC 2008

Authors:       Matthias Althoff
Written:       22-May-2020
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np


def tank6Eq(x, u):
    """
    tank6Eq - system dynamics for the tank benchmark
    
    Args:
        x: state vector (6-dimensional)
        u: input vector
        
    Returns:
        dx: time-derivative of the system state
    """
    # parameter
    k = 0.015
    k2 = 0.01
    g = 9.81
    
    # differential equations
    dx = np.zeros((6, 1))
    dx[0, 0] = u[0, 0] + 0.1 + k2 * (4 - x[5, 0]) - k * np.sqrt(2 * g) * np.sqrt(x[0, 0])  # tank 1
    dx[1, 0] = k * np.sqrt(2 * g) * (np.sqrt(x[0, 0]) - np.sqrt(x[1, 0]))  # tank 2
    dx[2, 0] = k * np.sqrt(2 * g) * (np.sqrt(x[1, 0]) - np.sqrt(x[2, 0]))  # tank 3
    dx[3, 0] = k * np.sqrt(2 * g) * (np.sqrt(x[2, 0]) - np.sqrt(x[3, 0]))  # tank 4
    dx[4, 0] = k * np.sqrt(2 * g) * (np.sqrt(x[3, 0]) - np.sqrt(x[4, 0]))  # tank 5
    dx[5, 0] = k * np.sqrt(2 * g) * (np.sqrt(x[4, 0]) - np.sqrt(x[5, 0]))  # tank 6
    
    return dx

