"""
minus - overloaded '-' operator for reachSet objects

Syntax:
    R = R1 - R2

Inputs:
    R1 - reachSet object
    R2 - reachSet object or numeric value

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

def minus(R1, R2):
    """
    Overloaded '-' operator for reachSet objects
    
    Args:
        R1: First reachSet object
        R2: Second reachSet object or numeric value
        
    Returns:
        Resulting reachSet object
    """
    
    # If R2 is numeric, subtract from all sets
    if isinstance(R2, (int, float, np.ndarray)):
        # Create new reachSet with shifted sets
        new_timePoint = {}
        new_timeInterval = {}
        
        # Shift time-point sets
        if 'set' in R1.timePoint:
            new_timePoint['set'] = []
            for s in R1.timePoint['set']:
                if hasattr(s, '__sub__'):
                    new_timePoint['set'].append(s - R2)
                elif isinstance(s, np.ndarray):
                    new_timePoint['set'].append(s - R2)
                else:
                    new_timePoint['set'].append(s)
            
            # Copy other fields
            for key in ['time', 'error']:
                if key in R1.timePoint:
                    new_timePoint[key] = R1.timePoint[key].copy()
        
        # Shift time-interval sets
        if 'set' in R1.timeInterval:
            new_timeInterval['set'] = []
            for s in R1.timeInterval['set']:
                if hasattr(s, '__sub__'):
                    new_timeInterval['set'].append(s - R2)
                elif isinstance(s, np.ndarray):
                    new_timeInterval['set'].append(s - R2)
                else:
                    new_timeInterval['set'].append(s)
            
            # Copy other fields
            for key in ['time', 'error', 'algebraic']:
                if key in R1.timeInterval:
                    new_timeInterval[key] = R1.timeInterval[key].copy()
        
        parent = R1.parent if hasattr(R1, 'parent') else 0
        loc = R1.loc if hasattr(R1, 'loc') else 0
        
        return ReachSet(new_timePoint, new_timeInterval, parent, loc)
    
    # If R2 is another reachSet, this is more complex (not commonly used)
    elif hasattr(R2, 'timePoint') and hasattr(R2, 'timeInterval'):
        # Basic implementation: subtract corresponding sets using Minkowski difference
        new_timePoint = {}
        new_timeInterval = {}
        
        # Handle time-point sets
        if 'set' in R1.timePoint and 'set' in R2.timePoint:
            new_timePoint['set'] = []
            min_len = min(len(R1.timePoint['set']), len(R2.timePoint['set']))
            
            for i in range(min_len):
                s1 = R1.timePoint['set'][i]
                s2 = R2.timePoint['set'][i]
                
                # Try Minkowski difference if available
                if hasattr(s1, '__sub__'):
                    new_timePoint['set'].append(s1 - s2)
                elif isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray):
                    new_timePoint['set'].append(s1 - s2)
                else:
                    # Fallback: just use first set
                    new_timePoint['set'].append(s1)
            
            # Copy time information from R1
            if 'time' in R1.timePoint:
                new_timePoint['time'] = R1.timePoint['time'][:min_len]
        
        # Handle time-interval sets similarly
        if 'set' in R1.timeInterval and 'set' in R2.timeInterval:
            new_timeInterval['set'] = []
            min_len = min(len(R1.timeInterval['set']), len(R2.timeInterval['set']))
            
            for i in range(min_len):
                s1 = R1.timeInterval['set'][i]
                s2 = R2.timeInterval['set'][i]
                
                if hasattr(s1, '__sub__'):
                    new_timeInterval['set'].append(s1 - s2)
                elif isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray):
                    new_timeInterval['set'].append(s1 - s2)
                else:
                    new_timeInterval['set'].append(s1)
            
            # Copy time information from R1
            if 'time' in R1.timeInterval:
                new_timeInterval['time'] = R1.timeInterval['time'][:min_len]
        
        parent = R1.parent if hasattr(R1, 'parent') else 0
        loc = R1.loc if hasattr(R1, 'loc') else 0
        
        return ReachSet(new_timePoint, new_timeInterval, parent, loc)
    
    else:
        raise TypeError(f"Unsupported operand type for -: reachSet and {type(R2)}") 