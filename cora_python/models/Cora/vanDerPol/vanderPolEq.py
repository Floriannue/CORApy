"""
vanderPolEq - system dynamics for the Van-der-Pol oscillator 
              (see Sec. VII in [1])

Syntax:
    dx = vanderPolEq(x, u)

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


def vanderPolEq(x, u):
    """
    vanderPolEq - system dynamics for the Van-der-Pol oscillator
    
    Args:
        x: state vector (2-dimensional)
        u: input vector
        
    Returns:
        dx: time-derivative of the system state
    """
    mu = 1
    
    dx = np.zeros((2, 1))
    dx[0, 0] = x[1, 0]
    dx[1, 0] = mu * (1 - x[0, 0]**2) * x[1, 0] - x[0, 0] + u[0, 0]
    
    return dx

