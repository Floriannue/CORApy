"""
eq - overloaded '==' operator for reachSet objects

Syntax:
    res = R1 == R2

Inputs:
    R1 - reachSet object
    R2 - reachSet object

Outputs:
    res - true if reachSet objects are equal, false otherwise

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""

import numpy as np


def eq(R1, R2) -> bool:
    """
    Overloaded '==' operator for reachSet objects
    
    Args:
        R1: First reachSet object
        R2: Second reachSet object
        
    Returns:
        True if reachSet objects are equal, False otherwise
    """
    # Check if both are reachSet objects
    if not (hasattr(R1, 'timePoint') and hasattr(R1, 'timeInterval')):
        return False
    if not (hasattr(R2, 'timePoint') and hasattr(R2, 'timeInterval')):
        return False
    
    # Compare parent and location
    if hasattr(R1, 'parent') and hasattr(R2, 'parent'):
        if R1.parent != R2.parent:
            return False
    elif hasattr(R1, 'parent') or hasattr(R2, 'parent'):
        return False
    
    if hasattr(R1, 'loc') and hasattr(R2, 'loc'):
        if R1.loc != R2.loc:
            return False
    elif hasattr(R1, 'loc') or hasattr(R2, 'loc'):
        return False
    
    # Compare time-point sets
    if not _compare_time_sets(R1.timePoint, R2.timePoint):
        return False
    
    # Compare time-interval sets
    if not _compare_time_sets(R1.timeInterval, R2.timeInterval):
        return False
    
    return True


def _compare_time_sets(set1, set2) -> bool:
    """Compare time-point or time-interval sets"""
    # Check if both have 'set' field
    has_set1 = 'set' in set1 and set1['set']
    has_set2 = 'set' in set2 and set2['set']
    
    if has_set1 != has_set2:
        return False
    
    if not has_set1:  # Both empty
        return True
    
    # Compare lengths
    if len(set1['set']) != len(set2['set']):
        return False
    
    # Compare each set
    for s1, s2 in zip(set1['set'], set2['set']):
        if hasattr(s1, '__eq__') and hasattr(s2, '__eq__'):
            if not (s1 == s2):
                return False
        elif isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray):
            if not np.array_equal(s1, s2):
                return False
        else:
            if s1 != s2:
                return False
    
    # Compare time fields
    if 'time' in set1 and 'time' in set2:
        if not np.array_equal(set1['time'], set2['time']):
            return False
    elif 'time' in set1 or 'time' in set2:
        return False
    
    # Compare error fields
    if 'error' in set1 and 'error' in set2:
        if not np.array_equal(set1['error'], set2['error']):
            return False
    elif 'error' in set1 or 'error' in set2:
        return False
    
    return True 