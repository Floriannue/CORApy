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
from typing import Tuple, Union


def priv_supportFunc(A: np.ndarray, b: np.ndarray, Ae: np.ndarray, be: np.ndarray, 
                     dir: np.ndarray, type: str) -> Tuple[float, Union[np.ndarray, None]]:
    """
    Computes the support function value for a polytope
    
    Args:
        A: Inequality constraint matrix
        b: Inequality constraint offset
        Ae: Equality constraint matrix
        be: Equality constraint offset
        dir: Direction vector
        type: 'upper' or 'lower'
        
    Returns:
        tuple: (val, x) where:
            val - value of the support function
            x - support vector (or None if infeasible/unbounded)
    """
    from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
    
    if type == 'upper':
        s = -1
    elif type == 'lower':
        s = 1
    else:
        raise ValueError("type must be 'upper' or 'lower'")
    
    # simple check: empty polytope (fullspace)
    if A.size == 0 and Ae.size == 0:
        val = -s * np.inf
        x = None
        return val, x
    
    # set up linear program
    # Ensure dir is a column vector and then transpose for f
    dir = dir.reshape(-1, 1)
    problem = {
        'f': (s * dir).flatten(),  # CORAlinprog expects 1D array
        'Aineq': A if A.size > 0 else None,
        'bineq': b.flatten() if A.size > 0 else None,
        'Aeq': Ae if Ae.size > 0 else None,
        'beq': be.flatten() if Ae.size > 0 else None,
        'lb': None,
        'ub': None
    }
    
    # solve linear program
    x, val, exitflag = CORAlinprog(problem)
    val = s * val
    
    if exitflag == -3:
        # unbounded
        val = -s * np.inf
        x = -s * np.sign(dir.flatten()) * np.inf * np.ones(len(dir))
    elif exitflag == -2:
        # infeasible -> empty set
        val = s * np.inf
        x = None
    elif exitflag != 1:
        raise CORAError('CORA:solverIssue', 'Linear programming solver failed')
    
    return val, x 