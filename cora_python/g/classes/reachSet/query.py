"""
query - get properties of the reachable set

Syntax:
    val = query(R, prop)

Inputs:
    R - reachSet object
    prop - property: 
           'reachSet': cell-array of time-interval solutions
           'reachSetTimePoint': cell-array of time-point solutions
           'finalSet': final time-point solution
           'tVec': vector of time steps
           'tFinal': final time stamp
           'allLoc': all location IDs of visited locations

Outputs:
    val - value of the property

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Last update: 10-April-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Any, Optional, List, Union


def query(R, prop: str) -> Any:
    """
    Get properties of the reachable set
    
    Args:
        R: reachSet object
        prop: Property to query
        
    Returns:
        Value of the requested property
    """
    # Validate inputs
    if not hasattr(R, 'timePoint') or not hasattr(R, 'timeInterval'):
        raise ValueError("Input must be a reachSet object")
    
    valid_props = ['reachSet', 'reachSetTimePoint', 'finalSet', 'tVec', 'tFinal', 'allLoc']
    if prop not in valid_props:
        raise ValueError(f"Property must be one of: {valid_props}")
    
    if prop == 'reachSet':
        # Return time-interval sets
        if 'set' in R.timeInterval:
            return R.timeInterval['set']
        else:
            return []
    
    elif prop == 'reachSetTimePoint':
        # Return time-point sets
        if 'set' in R.timePoint:
            return R.timePoint['set']
        else:
            return []
    
    elif prop == 'finalSet':
        # Return final time-point solution
        if 'set' in R.timePoint and len(R.timePoint['set']) > 0:
            return R.timePoint['set'][-1]
        else:
            return None
    
    elif prop == 'tVec':
        # Return vector of time steps
        if 'time' in R.timePoint:
            return np.array(R.timePoint['time'])
        elif 'time' in R.timeInterval:
            # For time intervals, return the start times
            times = R.timeInterval['time']
            if isinstance(times[0], (list, tuple, np.ndarray)) and len(times[0]) == 2:
                return np.array([t[0] for t in times])
            else:
                return np.array(times)
        else:
            return np.array([])
    
    elif prop == 'tFinal':
        # Return final time stamp
        if 'time' in R.timePoint and len(R.timePoint['time']) > 0:
            return R.timePoint['time'][-1]
        elif 'time' in R.timeInterval and len(R.timeInterval['time']) > 0:
            # For time intervals, return the end of the last interval
            last_time = R.timeInterval['time'][-1]
            if isinstance(last_time, (list, tuple, np.ndarray)) and len(last_time) == 2:
                return last_time[1]
            else:
                return last_time
        else:
            return 0.0
    
    elif prop == 'allLoc':
        # Return all location IDs
        if hasattr(R, 'loc'):
            if isinstance(R.loc, list):
                return list(set(R.loc))  # Remove duplicates
            else:
                return [R.loc]
        else:
            return [0]
    
    else:
        raise ValueError(f"Unknown property: {prop}") 