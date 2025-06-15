"""
add - add reachSet objects

Syntax:
    R = add(R1, R2)

Inputs:
    R1 - reachSet object
    R2 - reachSet object

Outputs:
    R - combined reachSet object

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""

import numpy as np


def add(R1, R2):
    """
    Add reachSet objects
    
    Args:
        R1: First reachSet object
        R2: Second reachSet object
        
    Returns:
        Combined reachSet object
    """
    from .reachSet import ReachSet
    
    # Validate inputs
    if not (hasattr(R1, 'timePoint') and hasattr(R1, 'timeInterval')):
        raise ValueError("R1 must be a reachSet object")
    if not (hasattr(R2, 'timePoint') and hasattr(R2, 'timeInterval')):
        raise ValueError("R2 must be a reachSet object")
    
    # Combine time-point sets
    combined_timePoint = {}
    if 'set' in R1.timePoint or 'set' in R2.timePoint:
        combined_timePoint['set'] = []
        combined_timePoint['time'] = []
        
        # Add sets from R1
        if 'set' in R1.timePoint:
            combined_timePoint['set'].extend(R1.timePoint['set'])
            combined_timePoint['time'].extend(R1.timePoint['time'])
        
        # Add sets from R2
        if 'set' in R2.timePoint:
            combined_timePoint['set'].extend(R2.timePoint['set'])
            combined_timePoint['time'].extend(R2.timePoint['time'])
        
        # Combine error fields if present
        if 'error' in R1.timePoint or 'error' in R2.timePoint:
            combined_timePoint['error'] = []
            if 'error' in R1.timePoint:
                combined_timePoint['error'].extend(R1.timePoint['error'])
            if 'error' in R2.timePoint:
                combined_timePoint['error'].extend(R2.timePoint['error'])
    
    # Combine time-interval sets
    combined_timeInterval = {}
    if 'set' in R1.timeInterval or 'set' in R2.timeInterval:
        combined_timeInterval['set'] = []
        combined_timeInterval['time'] = []
        
        # Add sets from R1
        if 'set' in R1.timeInterval:
            combined_timeInterval['set'].extend(R1.timeInterval['set'])
            combined_timeInterval['time'].extend(R1.timeInterval['time'])
        
        # Add sets from R2
        if 'set' in R2.timeInterval:
            combined_timeInterval['set'].extend(R2.timeInterval['set'])
            combined_timeInterval['time'].extend(R2.timeInterval['time'])
        
        # Combine error and algebraic fields if present
        for field in ['error', 'algebraic']:
            if field in R1.timeInterval or field in R2.timeInterval:
                combined_timeInterval[field] = []
                if field in R1.timeInterval:
                    combined_timeInterval[field].extend(R1.timeInterval[field])
                if field in R2.timeInterval:
                    combined_timeInterval[field].extend(R2.timeInterval[field])
    
    # Use parent and location from R1 (or could be combined differently)
    parent = R1.parent if hasattr(R1, 'parent') else 0
    loc = R1.loc if hasattr(R1, 'loc') else 0
    
    return ReachSet(combined_timePoint, combined_timeInterval, parent, loc) 