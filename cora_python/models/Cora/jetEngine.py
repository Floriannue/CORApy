"""
jetEngine - jet engine model from [1], Example 7

Syntax:
    dx = jetEngine(x, u)

Inputs:
    x - state vector
    u - input vector

Outputs:
    dx - time-derivative of the system state

References:
    [1] Under-approximate Flowpipes for Non-linear Continuous Systems
     Xin Chen, Sriram Sankaranarayanan, Erika Abraham
     ISBN: 978-0-9835678-4-4.

Authors:       Mark Wetzlinger
Written:       31-January-2020
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np


def jetEngine(x, u):
    """
    jetEngine - jet engine model dynamics
    
    Args:
        x: state vector (2-dimensional)
        u: input vector
        
    Returns:
        dx: time-derivative of the system state
    """
    dx = np.zeros((2, 1))
    dx[0, 0] = -x[1, 0] - 1.5 * x[0, 0]**2 - 0.5 * x[0, 0]**3 - 0.5
    dx[1, 0] = 3 * x[0, 0] - x[1, 0]
    
    return dx

