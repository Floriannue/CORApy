"""
isequal - checks if two reachSet objects are equal

Syntax:
    res = isequal(R1,R2)
    res = isequal(R1,R2,tol)

Inputs:
    R1 - reachSet object
    R2 - reachSet object
    tol - (optional) tolerance for set comparison

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: reachSet
"""

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .reachSet import ReachSet

def isequal(R1: 'ReachSet', R2: 'ReachSet', tol: float = 1e-12) -> bool:
    """
    Checks if two reachSet objects are equal.
    
    Args:
        R1: reachSet object
        R2: reachSet object
        tol: tolerance for set comparison (default: 1e-12)
        
    Returns:
        bool: True if reachSet objects are equal, False otherwise
    """
    # Validate inputs
    if tol < 0:
        raise ValueError("tolerance must be non-negative")
    
    # Handle single objects vs lists
    R1_list = R1 if isinstance(R1, list) else [R1]
    R2_list = R2 if isinstance(R2, list) else [R2]
    
    # check if same number of branches
    if len(R1_list) != len(R2_list):
        return False
    
    # loop over all branches: check if same parent/location
    for i in range(len(R1_list)):
        if hasattr(R1_list[i], 'parent') and hasattr(R2_list[i], 'parent'):
            if R1_list[i].parent != R2_list[i].parent:
                return False
        if hasattr(R1_list[i], 'loc') and hasattr(R2_list[i], 'loc'):
            if R1_list[i].loc != R2_list[i].loc:
                return False
    
    # check if each branch has same number of sets
    for i in range(len(R1_list)):
        # time-point solution
        R1empty = not R1_list[i].timePoint or not R1_list[i].timePoint.get('set')
        R2empty = not R2_list[i].timePoint or not R2_list[i].timePoint.get('set')
        if R1empty != R2empty:
            return False
        if not R1empty and not R2empty:
            if len(R1_list[i].timePoint['set']) != len(R2_list[i].timePoint['set']):
                return False
        
        # time-interval solution
        R1empty = not R1_list[i].timeInterval or not R1_list[i].timeInterval.get('set')
        R2empty = not R2_list[i].timeInterval or not R2_list[i].timeInterval.get('set')
        if R1empty != R2empty:
            return False
        if not R1empty and not R2empty:
            if len(R1_list[i].timeInterval['set']) != len(R2_list[i].timeInterval['set']):
                return False
    
    # check if time points and time intervals are equal
    for i in range(len(R1_list)):
        # time-point solution
        R1empty = not R1_list[i].timePoint or not R1_list[i].timePoint.get('set')
        if not R1empty:
            for j in range(len(R1_list[i].timePoint['time'])):
                t1 = R1_list[i].timePoint['time'][j]
                t2 = R2_list[i].timePoint['time'][j]
                if isinstance(t1, (int, float)) and isinstance(t2, (int, float)):
                    if not np.abs(t1 - t2) <= tol:
                        return False
                else:
                    # For interval objects or other types, use their isequal method if available
                    if hasattr(t1, 'isequal'):
                        if not t1.isequal(t2, tol):
                            return False
                    elif t1 != t2:
                        return False
        
        # time-interval solution
        R1empty = not R1_list[i].timeInterval or not R1_list[i].timeInterval.get('set')
        if not R1empty:
            for j in range(len(R1_list[i].timeInterval['time'])):
                t1 = R1_list[i].timeInterval['time'][j]
                t2 = R2_list[i].timeInterval['time'][j]
                if hasattr(t1, 'isequal'):
                    if not t1.isequal(t2, tol):
                        return False
                elif t1 != t2:
                    return False
    
    # check if sets are the same
    for i in range(len(R1_list)):
        # time-point solution
        R1empty = not R1_list[i].timePoint or not R1_list[i].timePoint.get('set')
        if not R1empty:
            # loop through sets
            for j in range(len(R1_list[i].timePoint['set'])):
                set1 = R1_list[i].timePoint['set'][j]
                set2 = R2_list[i].timePoint['set'][j]
                if hasattr(set1, 'isequal'):
                    if not set1.isequal(set2, tol):
                        return False
                elif set1 != set2:
                    return False
        
        # time-interval solution
        R1empty = not R1_list[i].timeInterval or not R1_list[i].timeInterval.get('set')
        if not R1empty:
            # loop through sets
            for j in range(len(R1_list[i].timeInterval['set'])):
                set1 = R1_list[i].timeInterval['set'][j]
                set2 = R2_list[i].timeInterval['set'][j]
                if hasattr(set1, 'isequal'):
                    if not set1.isequal(set2, tol):
                        return False
                elif set1 != set2:
                    return False
    
    return True 