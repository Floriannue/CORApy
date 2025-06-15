"""
append - appends one reachSet object at the end of another one
    (currently, only one branch supported)

Syntax:
    obj = append(R, Radd)

Inputs:
    R - reachSet object
    Radd - reachSet object

Outputs:
    R - resulting reachSet object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: reachSet
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .reachSet import ReachSet

def append(R: 'ReachSet', Radd: 'ReachSet') -> 'ReachSet':
    """
    Appends one reachSet object at the end of another one.
    Currently, only one branch per reachSet object is supported.
    
    Args:
        R: reachSet object
        Radd: reachSet object to append
        
    Returns:
        ReachSet: resulting reachSet object
        
    Raises:
        ValueError: if multiple branches are provided (not supported)
    """
    # currently, only one branch per reachSet object
    if len(R) > 1 or len(Radd) > 1:
        raise ValueError('Multiple branches not supported.')
    
    # empty objects
    if R.isemptyobject():
        return Radd
    elif Radd.isemptyobject():
        return R
    else:
        # general case:
        
        # get final time of R
        shift = R.timePoint.time[-1]
        
        # number of sets of Radd
        nrSets_tp = len(Radd.timePoint.set)
        nrSets_ti = len(Radd.timeInterval.set) if Radd.timeInterval.set is not None else 0
        
        # time-point sets
        R.timePoint.set.extend(Radd.timePoint.set)
        # shift time
        shifted_times = []
        for i in range(nrSets_tp):
            shifted_times.append(Radd.timePoint.time[i] + shift)
        R.timePoint.time.extend(shifted_times)
        
        # time-interval sets
        if R.timeInterval.set is not None and Radd.timeInterval.set is not None:
            R.timeInterval.set.extend(Radd.timeInterval.set)
            # shift time
            shifted_times_ti = []
            for i in range(nrSets_ti):
                shifted_times_ti.append(Radd.timeInterval.time[i] + shift)
            R.timeInterval.time.extend(shifted_times_ti)
    
    return R 