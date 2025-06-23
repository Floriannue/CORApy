"""
mtimes - overloaded '*' operator for simResult objects (matrix multiplication)

Syntax:
    simRes = simRes * matrix
    simRes = matrix * simRes

Inputs:
    simRes - simResult object
    matrix - numeric matrix

Outputs:
    simRes - resulting simResult object

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""

import numpy as np
from .simResult import SimResult

def mtimes(arg1, arg2):
    """
    Overloaded '*' operator for simResult objects (matrix multiplication)
    
    Args:
        arg1: simResult object or numeric matrix
        arg2: simResult object or numeric matrix
        
    Returns:
        Resulting simResult object
    """
    
    # Determine which argument is the simResult and which is the matrix
    if hasattr(arg1, 'x') and hasattr(arg1, 't'):
        simRes = arg1
        matrix = np.asarray(arg2)
        left_multiply = False
    elif hasattr(arg2, 'x') and hasattr(arg2, 't'):
        simRes = arg2
        matrix = np.asarray(arg1)
        left_multiply = True
    else:
        raise TypeError("One argument must be a simResult object")
    
    # Apply matrix multiplication to states
    if left_multiply:
        # matrix * simRes
        new_x = [matrix @ x.T for x in simRes.x]
        new_x = [x.T for x in new_x]  # Transpose back
    else:
        # simRes * matrix
        new_x = [x @ matrix.T for x in simRes.x]
    
    # For outputs, only apply transformation if dimensions match
    new_y = []
    if simRes.y:
        for y in simRes.y:
            if y.size > 0:
                try:
                    if left_multiply:
                        if y.shape[1] == matrix.shape[1]:  # Compatible dimensions
                            new_y.append((matrix @ y.T).T)
                        else:
                            new_y.append(y.copy())  # Keep original if incompatible
                    else:
                        if y.shape[1] == matrix.shape[0]:  # Compatible dimensions
                            new_y.append(y @ matrix.T)
                        else:
                            new_y.append(y.copy())  # Keep original if incompatible
                except ValueError:
                    # If matrix multiplication fails due to dimension mismatch, keep original
                    new_y.append(y.copy())
            else:
                new_y.append(y.copy())
    
    # Algebraic variables are typically not transformed
    new_a = [a.copy() for a in simRes.a] if simRes.a else []
    
    return SimResult(new_x, simRes.t.copy(), simRes.loc, new_y, new_a) 