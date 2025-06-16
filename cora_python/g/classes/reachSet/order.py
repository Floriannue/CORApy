"""
order - orders the elements of a reachSet object chronologically;
    currently only supported for one branch

Syntax:
    R = order(R)

Inputs:
    R - reachSet object

Outputs:
    R - reachSet object

Example: 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none
"""

from typing import TYPE_CHECKING, Union
import numpy as np

if TYPE_CHECKING:
    from .reachSet import ReachSet

def order(R: 'ReachSet'):
    """
    Orders the elements of a reachSet object chronologically.
    Currently only supported for one branch.
    
    If the reachSet contains sets with an order method (like zonotopes),
    returns the order of the first set. Otherwise, returns the ordered reachSet.
    
    Args:
        R: reachSet object
        
    Returns:
        Union[int, float, ReachSet]: order of first set or ordered reachSet object
        
    Raises:
        ValueError: if multiple branches are provided (not supported)
    """
    # Handle single object vs list
    R_list = R if isinstance(R, list) else [R]
    
    # only supported for one branch
    if len(R_list) > 1:
        raise ValueError('Only supports one branch.')
    
    R_obj = R_list[0]
    
    # Check if we should return the order of the first set
    if (R_obj.timePoint and 'set' in R_obj.timePoint and 
        R_obj.timePoint['set'] and len(R_obj.timePoint['set']) > 0):
        first_set = R_obj.timePoint['set'][0]
        # If the set has a G attribute (like zonotope), return its order (number of generators)
        if hasattr(first_set, 'G') and hasattr(first_set.G, 'shape'):
            return first_set.G.shape[1] if first_set.G.size > 0 else 0
    
    # Otherwise, perform chronological ordering and return the reachSet
    
    # time-point solutions
    if R_obj.timePoint and 'time' in R_obj.timePoint and R_obj.timePoint['time']:
        time_array = np.array(R_obj.timePoint['time'])
        sorted_indices = np.argsort(time_array)
        
        # Reorder sets and times
        R_obj.timePoint['set'] = [R_obj.timePoint['set'][i] for i in sorted_indices]
        R_obj.timePoint['time'] = [R_obj.timePoint['time'][i] for i in sorted_indices]
        
        # error only for linearSys
        if 'error' in R_obj.timePoint and R_obj.timePoint['error']:
            R_obj.timePoint['error'] = [R_obj.timePoint['error'][i] for i in sorted_indices]
    
    # time-interval solutions
    if (R_obj.timeInterval and 'time' in R_obj.timeInterval and 
        R_obj.timeInterval['time'] and len(R_obj.timeInterval['time']) > 0):
        
        time_intervals = R_obj.timeInterval['time']
        nrSets = len(time_intervals)
        
        # read out interval bounds
        timeInf = np.zeros(nrSets)
        timeSup = np.zeros(nrSets)
        
        for i in range(nrSets):
            time_interval = time_intervals[i]
            if hasattr(time_interval, 'infimum') and hasattr(time_interval, 'supremum'):
                timeInf[i] = time_interval.infimum
                timeSup[i] = time_interval.supremum
            elif hasattr(time_interval, 'inf') and hasattr(time_interval, 'sup'):
                timeInf[i] = time_interval.inf
                timeSup[i] = time_interval.sup
            elif isinstance(time_interval, (list, tuple)) and len(time_interval) == 2:
                timeInf[i] = time_interval[0]
                timeSup[i] = time_interval[1]
            else:
                # Assume it's a single time point
                timeInf[i] = float(time_interval)
                timeSup[i] = float(time_interval)
        
        # order depending on start time of interval
        sorted_indices = np.argsort(timeInf)
        timeInf = timeInf[sorted_indices]
        timeSup = timeSup[sorted_indices]
        
        # new list of sets
        R_obj.timeInterval['set'] = [R_obj.timeInterval['set'][i] for i in sorted_indices]
        
        # new list of time intervals
        new_time_intervals = []
        for i in range(nrSets):
            # Try to create interval object if available
            try:
                from ...contSet.interval.interval import Interval
                new_time_intervals.append(Interval(timeInf[i], timeSup[i]))
            except ImportError:
                # Fallback to tuple representation
                new_time_intervals.append((timeInf[i], timeSup[i]))
        
        R_obj.timeInterval['time'] = new_time_intervals
        
        # error only for linearSys
        if 'error' in R_obj.timeInterval and R_obj.timeInterval['error']:
            R_obj.timeInterval['error'] = [R_obj.timeInterval['error'][i] for i in sorted_indices]
    
    return R_obj if not isinstance(R, list) else R_list 