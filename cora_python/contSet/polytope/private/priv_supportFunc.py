"""
priv_supportFunc - computes the support function value for a polytope

Syntax:
    val, x = priv_supportFunc(A, b, Ae, be, dir, type)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    dir - direction
    type - 'upper' or 'lower'

Outputs:
    val - value of the support function
    x - support vector

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Optional
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_supportFunc(
    A: np.ndarray, b: np.ndarray,
    Ae: np.ndarray, be: np.ndarray,
    dir_vec: np.ndarray, type_str: str
) -> Tuple[float, Optional[np.ndarray]]:
    """
    priv_supportFunc - computes the support function of a polytope in H-representation
    
    Syntax:
       [val,x] = priv_supportFunc(A,b,Ae,be,dir,type)
    
    Inputs:
       A - inequality constraint matrix
       b - inequality constraint offset
       Ae - equality constraint matrix
       be - equality constraint offset
       dir - direction
       type - 'upper' or 'lower'
    
    Outputs:
       val - value of the support function
       x - support vector
    
    Authors:       Mark Wetzlinger (MATLAB)
                   Automatic python translation: Florian Nüssel BA 2025
    Written:       03-October-2024 (MATLAB)
    Last update:   ---
    Last revision: ---
    """

    s = 0.0
    if type_str == 'upper':
        s = -1.0
    elif type_str == 'lower':
        s = 1.0
    else:
        raise ValueError("Type must be 'upper' or 'lower'.")

    # simple check: empty polytope (fullspace in MATLAB original doc, but refers to no constraints)
    # In Python, empty arrays will be (0,0) or (0,1) shape
    if (A.size == 0 and b.size == 0) and (Ae.size == 0 and be.size == 0):
        val = -s * np.inf
        x = np.array([]) # Return an empty numpy array for x
        return val, x

    # set up linear program
    problem = {
        'f': s * dir_vec.flatten(), # Flatten dir to 1D array for linprog
        'Aineq': A,
        'bineq': b.flatten(), # Flatten b to 1D array
        'Aeq': Ae,
        'beq': be.flatten(), # Flatten be to 1D array
        'lb': None,
        'ub': None,
        'x0': None # Not used in scipy linprog directly
    }

    # solve linear program
    x_sol, fval, exitflag, output, _ = CORAlinprog(problem)
    val = s * fval if fval is not None else None

    if exitflag == -3:
        # unbounded
        val = -s * np.inf
        # Handle x: -s*sign(dir).*Inf(length(dir),1)
        # dir_vec must be 1D, so length(dir) is dir_vec.size
        x = -s * np.sign(dir_vec) * np.inf
        if x.ndim == 1: # Ensure it's a column vector if it was 1D
            x = x.reshape(-1, 1)
    elif exitflag == -2:
        # infeasible -> empty set
        val = s * np.inf
        x = np.array([]) # Empty array as per MATLAB
    elif exitflag != 1:
        raise CORAerror('CORA:solverIssue', f"Solver issue with exitflag: {exitflag}, message: {output.get('message', '')}")
    
    # Ensure x is a column vector if it's a solution and not None
    if x_sol is not None and x is None: # If x was not set by unbounded/infeasible and x_sol exists
        x = x_sol.reshape(-1, 1)

    return val, x 