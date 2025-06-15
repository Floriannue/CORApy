"""
mtimes - overloaded '*' operator for reachSet objects (matrix multiplication)

Syntax:
    R = R1 * R2

Inputs:
    R1 - reachSet object or numeric matrix
    R2 - reachSet object or numeric matrix

Outputs:
    R - resulting reachSet object

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""

import numpy as np


def mtimes(arg1, arg2):
    """
    Overloaded '*' operator for reachSet objects (matrix multiplication)
    
    Args:
        arg1: reachSet object or numeric matrix
        arg2: reachSet object or numeric matrix
        
    Returns:
        Resulting reachSet object
    """
    from .reachSet import ReachSet
    
    # Determine which argument is the reachSet and which is the matrix
    if hasattr(arg1, 'timePoint') and hasattr(arg1, 'timeInterval'):
        R = arg1
        matrix = np.asarray(arg2)
        left_multiply = False
    elif hasattr(arg2, 'timePoint') and hasattr(arg2, 'timeInterval'):
        R = arg2
        matrix = np.asarray(arg1)
        left_multiply = True
    else:
        raise TypeError("One argument must be a reachSet object")
    
    # Create new reachSet with transformed sets
    new_timePoint = {}
    new_timeInterval = {}
    
    # Transform time-point sets
    if 'set' in R.timePoint:
        new_timePoint['set'] = []
        for s in R.timePoint['set']:
            if hasattr(s, '__matmul__') or hasattr(s, 'dot'):
                if left_multiply:
                    # matrix * set
                    if hasattr(s, '__rmatmul__'):
                        new_timePoint['set'].append(matrix @ s)
                    else:
                        new_timePoint['set'].append(np.dot(matrix, s))
                else:
                    # set * matrix
                    if hasattr(s, '__matmul__'):
                        new_timePoint['set'].append(s @ matrix)
                    else:
                        new_timePoint['set'].append(np.dot(s, matrix))
            elif isinstance(s, np.ndarray):
                if left_multiply:
                    new_timePoint['set'].append(matrix @ s)
                else:
                    new_timePoint['set'].append(s @ matrix)
            else:
                new_timePoint['set'].append(s)
        
        # Copy other fields
        for key in ['time', 'error']:
            if key in R.timePoint:
                new_timePoint[key] = R.timePoint[key].copy()
    
    # Transform time-interval sets
    if 'set' in R.timeInterval:
        new_timeInterval['set'] = []
        for s in R.timeInterval['set']:
            if hasattr(s, '__matmul__') or hasattr(s, 'dot'):
                if left_multiply:
                    # matrix * set
                    if hasattr(s, '__rmatmul__'):
                        new_timeInterval['set'].append(matrix @ s)
                    else:
                        new_timeInterval['set'].append(np.dot(matrix, s))
                else:
                    # set * matrix
                    if hasattr(s, '__matmul__'):
                        new_timeInterval['set'].append(s @ matrix)
                    else:
                        new_timeInterval['set'].append(np.dot(s, matrix))
            elif isinstance(s, np.ndarray):
                if left_multiply:
                    new_timeInterval['set'].append(matrix @ s)
                else:
                    new_timeInterval['set'].append(s @ matrix)
            else:
                new_timeInterval['set'].append(s)
        
        # Copy other fields
        for key in ['time', 'error', 'algebraic']:
            if key in R.timeInterval:
                new_timeInterval[key] = R.timeInterval[key].copy()
    
    parent = R.parent if hasattr(R, 'parent') else 0
    loc = R.loc if hasattr(R, 'loc') else 0
    
    return ReachSet(new_timePoint, new_timeInterval, parent, loc) 