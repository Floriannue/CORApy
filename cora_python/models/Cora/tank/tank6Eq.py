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
import sympy as sp


def tank6Eq(x, u):
    """
    tank6Eq - system dynamics for the tank benchmark
    
    Args:
        x: state vector (6-dimensional) - can be numpy array or sympy Matrix
        u: input vector - can be numpy array or sympy Matrix
        
    Returns:
        dx: time-derivative of the system state (same type as input)
    """
    # parameter
    k = 0.015
    k2 = 0.01
    g = 9.81
    
    # Check if inputs are symbolic (sympy Matrix)
    is_symbolic = isinstance(x, (sp.Matrix, sp.Basic)) or isinstance(u, (sp.Matrix, sp.Basic))
    
    if is_symbolic:
        # Use sympy operations for symbolic computation
        # Create symbolic zero matrix
        dx = sp.zeros(6, 1)
        # For sympy Matrix, use single index for column vectors: x[i] instead of x[i, 0]
        dx[0, 0] = u[0, 0] + 0.1 + k2 * (4 - x[5, 0]) - k * sp.sqrt(2 * g) * sp.sqrt(x[0, 0])  # tank 1
        dx[1, 0] = k * sp.sqrt(2 * g) * (sp.sqrt(x[0, 0]) - sp.sqrt(x[1, 0]))  # tank 2
        dx[2, 0] = k * sp.sqrt(2 * g) * (sp.sqrt(x[1, 0]) - sp.sqrt(x[2, 0]))  # tank 3
        dx[3, 0] = k * sp.sqrt(2 * g) * (sp.sqrt(x[2, 0]) - sp.sqrt(x[3, 0]))  # tank 4
        dx[4, 0] = k * sp.sqrt(2 * g) * (sp.sqrt(x[3, 0]) - sp.sqrt(x[4, 0]))  # tank 5
        dx[5, 0] = k * sp.sqrt(2 * g) * (sp.sqrt(x[4, 0]) - sp.sqrt(x[5, 0]))  # tank 6
    else:
        # Use numpy operations for numeric computation
        dx = np.zeros((6, 1))
        dx[0, 0] = u[0, 0] + 0.1 + k2 * (4 - x[5, 0]) - k * np.sqrt(2 * g) * np.sqrt(x[0, 0])  # tank 1
        dx[1, 0] = k * np.sqrt(2 * g) * (np.sqrt(x[0, 0]) - np.sqrt(x[1, 0]))  # tank 2
        dx[2, 0] = k * np.sqrt(2 * g) * (np.sqrt(x[1, 0]) - np.sqrt(x[2, 0]))  # tank 3
        dx[3, 0] = k * np.sqrt(2 * g) * (np.sqrt(x[2, 0]) - np.sqrt(x[3, 0]))  # tank 4
        dx[4, 0] = k * np.sqrt(2 * g) * (np.sqrt(x[3, 0]) - np.sqrt(x[4, 0]))  # tank 5
        dx[5, 0] = k * np.sqrt(2 * g) * (np.sqrt(x[4, 0]) - np.sqrt(x[5, 0]))  # tank 6
    
    return dx

