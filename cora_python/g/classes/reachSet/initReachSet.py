"""
initReachSet - create an object of class reachSet that stores the reachable set

Syntax:
    R = initReachSet(timePoint, timeInt)

Inputs:
    timePoint - dict containing the time-point reachable set
    timeInt - dict containing the time-interval reachable set

Outputs:
    R - reachSet object

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 19-November-2022 (MATLAB)
Last update: ---
Python translation: 2025
"""

import numpy as np
from typing import Dict, Optional


def initReachSet(timePoint: Dict, timeInt: Optional[Dict] = None):
    """
    Create an object of class reachSet that stores the reachable set
    
    Args:
        timePoint: Dict containing the time-point reachable set
        timeInt: Dict containing the time-interval reachable set (optional)
        
    Returns:
        reachSet object
    """
    from .reachSet import ReachSet
    
    # Handle case where no time-interval solution is provided
    if timeInt is None:
        timeInt = {}
    
    # Remove empty cells from the end (occurs due to premature exit)
    def is_empty_set(x):
        """Check if a set is empty"""
        if x is None:
            return True
        if hasattr(x, 'isemptyobject') and x.isemptyobject():
            return True
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 0:
            return True
        return False
    
    # Clean up timeInt
    if 'set' in timeInt and timeInt['set']:
        # Remove empty sets from the end
        while timeInt['set'] and is_empty_set(timeInt['set'][-1]):
            timeInt['set'].pop()
            if 'time' in timeInt and timeInt['time']:
                timeInt['time'].pop()
            if 'algebraic' in timeInt and timeInt['algebraic']:
                timeInt['algebraic'].pop()
            if 'error' in timeInt and timeInt['error']:
                timeInt['error'].pop()
    
    # Clean up timePoint
    if 'set' in timePoint and timePoint['set']:
        # Remove empty sets from the end
        while timePoint['set'] and is_empty_set(timePoint['set'][-1]):
            timePoint['set'].pop()
            if 'time' in timePoint and timePoint['time']:
                timePoint['time'].pop()
            if 'error' in timePoint and timePoint['error']:
                timePoint['error'].pop()
    
    # Check if the first time-point set is empty
    if ('set' in timePoint and timePoint['set'] and 
        hasattr(timePoint['set'][0], 'representsa_') and 
        timePoint['set'][0].representsa_('emptySet')):
        # Create reachSet with empty time-point
        return ReachSet({}, timeInt)
    
    # Create the reachSet object
    return ReachSet(timePoint, timeInt) 