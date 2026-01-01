"""
laubLoomis - dynamic equation for the Laub-Loomis benchmark 
             (see Sec. 3.2 in [1])

Syntax:
    dx = laubLoomis(x, u)

Inputs:
    x - state vector
    u - input vector

Outputs:
    dx - time-derivative of the system state

References:
    [1] F. Immler, "ARCH-COMP19 Category Report: Continuous and Hybrid 
        Systems with Nonlinear Dynamics", 2019

Authors:       Niklas Kochdumper
Written:       19-June-2020
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np


def laubLoomis(x, u):
    """
    laubLoomis - dynamic equation for the Laub-Loomis benchmark
    
    Args:
        x: state vector (7-dimensional)
        u: input vector
        
    Returns:
        dx: time-derivative of the system state
    """
    dx = np.zeros((7, 1))
    dx[0, 0] = 1.4 * x[2, 0] - 0.9 * x[0, 0]
    dx[1, 0] = 2.5 * x[4, 0] - 1.5 * x[1, 0]
    dx[2, 0] = 0.6 * x[6, 0] - 0.8 * x[2, 0] * x[1, 0]
    dx[3, 0] = 2.0 - 1.3 * x[3, 0] * x[2, 0]
    dx[4, 0] = 0.7 * x[0, 0] - 1.0 * x[3, 0] * x[4, 0]
    dx[5, 0] = 0.3 * x[0, 0] - 3.1 * x[5, 0]
    dx[6, 0] = 1.8 * x[5, 0] - 1.5 * x[6, 0] * x[1, 0]
    
    return dx

