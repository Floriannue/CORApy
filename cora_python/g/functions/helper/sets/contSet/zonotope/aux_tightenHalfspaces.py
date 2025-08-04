"""
AUX_TIGHTENHALFSPACES - tighten halfspaces so that the polytope is identical with the same number of halfspaces

    d_new = aux_tightenHalfspaces(C, delta_d)

    Inputs:
        C - constraint matrix
        delta_d - constraint vector

    Outputs:
        d_new - tightened constraint vector

    Example:
        C = [1 0; -1 0; 0 1; 0 -1];
        delta_d = [1; 1; 1; 1];
        d_new = aux_tightenHalfspaces(C, delta_d);

    Author:        Florian Nüssel
    Written:       ---
    Last update:   ---
    Last revision: ---

    Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Optional
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog


def aux_tightenHalfspaces(C: np.ndarray, delta_d: np.ndarray) -> Optional[np.ndarray]:
    """
    Tighten halfspaces so that the polytope is identical with the same number of halfspaces
    
    Args:
        C: constraint matrix
        delta_d: constraint vector
        
    Returns:
        d_new: tightened constraint vector, or None if polytope is empty
    """
    # Init linprog struct
    problem = {
        'Aineq': C,
        'bineq': delta_d,
        'Aeq': None,
        'beq': None,
        'lb': None,
        'ub': None
    }
    
    # Loop over halfspaces
    d_new = np.zeros((len(delta_d), 1))
    for i in range(len(delta_d)):
        # Normal vector
        problem['f'] = -C[i, :]
        _, d_new[i, 0], exitflag = CORAlinprog(problem)[:3]
    
    if exitflag != 1:
        # Linear program is infeasible since polytope is empty
        return None
    else:
        # Values have the opposite sign
        return -d_new 