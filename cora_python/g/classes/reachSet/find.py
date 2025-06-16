"""
find - get reachSet object that satisfy a certain condition

Syntax:
    res = find(R, prop, val)

Inputs:
    R - reachSet object
    prop - property for condition ('location', 'parent', 'time')
    val - value for property

Outputs:
    res - all reachSet objects that satisfy the condition

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""

import numpy as np
from typing import Any, Optional, Union


def find(R, prop: str, val: Any):
    """
    Get reachSet objects that satisfy a certain condition
    
    Args:
        R: reachSet object or list of reachSet objects
        prop: Property for condition ('location', 'parent', 'time')
        val: Value for property
        
    Returns:
        reachSet objects that satisfy the condition
    """
    # Validate inputs
    valid_props = ['location', 'parent', 'time']
    if prop not in valid_props:
        raise ValueError(f"Property must be one of: {valid_props}")
    
    # Handle single reachSet object
    if not isinstance(R, list):
        R_list = [R]
    else:
        R_list = R
    
    # Initialize result
    result = []
    
    if prop == 'location':
        # Find reachSet objects with specified location
        for r in R_list:
            if hasattr(r, 'loc') and r.loc == val:
                result.append(r)
    
    elif prop == 'parent':
        # Find reachSet objects with specified parent
        for r in R_list:
            if hasattr(r, 'parent') and r.parent == val:
                result.append(r)
    
    elif prop == 'time':
        # Find reachSet objects inside the specified time interval
        from ....contSet.interval.interval import Interval
        
        # Convert val to interval if it's not already
        if not isinstance(val, Interval):
            if isinstance(val, (int, float)):
                # Single time point - create small interval around it
                val = Interval(val - 1e-10, val + 1e-10)
            elif isinstance(val, (list, tuple, np.ndarray)) and len(val) == 2:
                val = Interval(val[0], val[1])
            else:
                raise ValueError("Time value must be a number, interval, or [start, end] pair")
        
        for r in R_list:
            # Check time-interval sets
            if 'time' in r.timeInterval and 'set' in r.timeInterval:
                times = r.timeInterval['time']
                sets = r.timeInterval['set']
                
                for i, (time_int, set_obj) in enumerate(zip(times, sets)):
                    # Convert time_int to interval if needed
                    if isinstance(time_int, (list, tuple, np.ndarray)) and len(time_int) == 2:
                        t_int = Interval(time_int[0], time_int[1])
                    else:
                        t_int = Interval(time_int, time_int)
                    
                    # Check if intervals overlap
                    if _intervals_overlap(val, t_int):
                        # Create new reachSet with only the overlapping part
                        new_timeInterval = {
                            'set': [set_obj],
                            'time': [time_int]
                        }
                        if 'error' in r.timeInterval:
                            new_timeInterval['error'] = [r.timeInterval['error'][i]]
                        
                        from .reachSet import ReachSet
                        new_r = ReachSet(timeInterval=new_timeInterval, parent=r.parent, loc=r.loc)
                        result.append(new_r)
            
            # Check time-point sets
            if 'time' in r.timePoint and 'set' in r.timePoint:
                times = r.timePoint['time']
                sets = r.timePoint['set']
                
                for i, (time_pt, set_obj) in enumerate(zip(times, sets)):
                    # Check if time point is in interval
                    if val.inf <= time_pt <= val.sup:
                        # Create new reachSet with only this time point
                        new_timePoint = {
                            'set': [set_obj],
                            'time': [time_pt]
                        }
                        if 'error' in r.timePoint:
                            new_timePoint['error'] = [r.timePoint['error'][i]]
                        
                        from .reachSet import ReachSet
                        new_r = ReachSet(timePoint=new_timePoint, parent=r.parent, loc=r.loc)
                        result.append(new_r)
    
    # Return result
    if len(result) == 0:
        return None
    elif len(result) == 1:
        return result[0]
    else:
        return result


def _intervals_overlap(int1, int2) -> bool:
    """Check if two intervals overlap"""
    return not (int1.sup < int2.inf or int2.sup < int1.inf) 