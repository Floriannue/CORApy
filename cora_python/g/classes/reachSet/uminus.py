"""
uminus - overloaded unary '-' operator for reachSet objects

Syntax:
    R = -R1

Inputs:
    R1 - reachSet object

Outputs:
    R - resulting reachSet object

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""

import numpy as np
from .reachSet import ReachSet

def uminus(R1):
    """
    Overloaded unary '-' operator for reachSet objects
    
    Args:
        R1: reachSet object
        
    Returns:
        Resulting reachSet object with negated sets
    """
    
    # Create new reachSet with negated sets
    new_timePoint = {}
    new_timeInterval = {}
    
    # Negate time-point sets
    if 'set' in R1.timePoint:
        new_timePoint['set'] = []
        for s in R1.timePoint['set']:
            if hasattr(s, '__neg__'):
                new_timePoint['set'].append(-s)
            elif isinstance(s, np.ndarray):
                new_timePoint['set'].append(-s)
            else:
                new_timePoint['set'].append(s)
        
        # Copy other fields
        for key in ['time', 'error']:
            if key in R1.timePoint:
                new_timePoint[key] = R1.timePoint[key].copy()
    
    # Negate time-interval sets
    if 'set' in R1.timeInterval:
        new_timeInterval['set'] = []
        for s in R1.timeInterval['set']:
            if hasattr(s, '__neg__'):
                new_timeInterval['set'].append(-s)
            elif isinstance(s, np.ndarray):
                new_timeInterval['set'].append(-s)
            else:
                new_timeInterval['set'].append(s)
        
        # Copy other fields
        for key in ['time', 'error', 'algebraic']:
            if key in R1.timeInterval:
                new_timeInterval[key] = R1.timeInterval[key].copy()
    
    parent = R1.parent if hasattr(R1, 'parent') else 0
    loc = R1.loc if hasattr(R1, 'loc') else 0
    
    return ReachSet(new_timePoint, new_timeInterval, parent, loc) 