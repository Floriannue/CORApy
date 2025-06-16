"""
shiftTime - shifts all sets of a reachSet object by a scalar time delta

Syntax:
    R = shiftTime(R,delta)

Inputs:
    R - reachSet object 
    delta - scalar

Outputs:
    R - shifted reachset object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: plus
"""

from typing import TYPE_CHECKING, Union
import numpy as np

if TYPE_CHECKING:
    from .reachSet import ReachSet

def shiftTime(R: 'ReachSet', delta: Union[int, float]) -> 'ReachSet':
    """
    Shifts all sets of a reachSet object by a scalar time delta.
    
    Args:
        R: reachSet object
        delta: scalar time shift
        
    Returns:
        ReachSet: shifted reachSet object
    """
    # Validate inputs
    if not isinstance(delta, (int, float)):
        raise ValueError("delta must be a scalar number")
    
    # Handle single object vs list
    R_list = R if isinstance(R, list) else [R]
    
    # shift time vector
    for i in range(len(R_list)):
        R_obj = R_list[i]
        
        # Shift time-interval sets
        if (R_obj.timeInterval and 'time' in R_obj.timeInterval and 
            R_obj.timeInterval['time']):
            new_time_intervals = []
            for time_interval in R_obj.timeInterval['time']:
                if hasattr(time_interval, '__add__'):
                    # If the time interval supports addition
                    new_time_intervals.append(delta + time_interval)
                elif hasattr(time_interval, 'infimum') and hasattr(time_interval, 'supremum'):
                    # Interval object with infimum/supremum
                    try:
                        from ...contSet.interval.interval import Interval
                        new_time_intervals.append(Interval(
                            time_interval.infimum + delta,
                            time_interval.supremum + delta
                        ))
                    except ImportError:
                        new_time_intervals.append((
                            time_interval.infimum + delta,
                            time_interval.supremum + delta
                        ))
                elif isinstance(time_interval, (list, tuple)) and len(time_interval) == 2:
                    # Tuple/list representation
                    new_time_intervals.append((
                        time_interval[0] + delta,
                        time_interval[1] + delta
                    ))
                else:
                    # Scalar time
                    new_time_intervals.append(time_interval + delta)
            
            R_obj.timeInterval['time'] = new_time_intervals
        
        # Shift time-point sets
        if (R_obj.timePoint and 'time' in R_obj.timePoint and 
            R_obj.timePoint['time']):
            new_time_points = []
            for time_point in R_obj.timePoint['time']:
                if hasattr(time_point, '__add__'):
                    # If the time point supports addition
                    new_time_points.append(delta + time_point)
                else:
                    # Scalar time
                    new_time_points.append(time_point + delta)
            
            R_obj.timePoint['time'] = new_time_points
    
    return R_list[0] if not isinstance(R, list) else R_list 