"""
uplus - overloaded unary '+' operator for reachSet objects

Syntax:
    R = +R1

Inputs:
    R1 - reachSet object

Outputs:
    R - resulting reachSet object (unchanged)

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""
from .reachSet import ReachSet

def uplus(R1):
    """
    Overloaded unary '+' operator for reachSet objects
    
    Args:
        R1: reachSet object
        
    Returns:
        Resulting reachSet object (unchanged copy)
    """
    
    # Create a copy of the reachSet (unary plus doesn't change anything)
    new_timePoint = {}
    new_timeInterval = {}
    
    # Copy time-point sets
    if 'set' in R1.timePoint:
        new_timePoint['set'] = R1.timePoint['set'].copy()
        # Copy other fields
        for key in ['time', 'error']:
            if key in R1.timePoint:
                new_timePoint[key] = R1.timePoint[key].copy()
    
    # Copy time-interval sets
    if 'set' in R1.timeInterval:
        new_timeInterval['set'] = R1.timeInterval['set'].copy()
        # Copy other fields
        for key in ['time', 'error', 'algebraic']:
            if key in R1.timeInterval:
                new_timeInterval[key] = R1.timeInterval[key].copy()
    
    parent = R1.parent if hasattr(R1, 'parent') else 0
    loc = R1.loc if hasattr(R1, 'loc') else 0
    
    return ReachSet(new_timePoint, new_timeInterval, parent, loc) 