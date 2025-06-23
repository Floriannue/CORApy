"""
reachSet - class for storing reachable sets

This class stores reachable sets computed by reachability analysis algorithms.
It contains both time-point and time-interval reachable sets.

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: 15-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union


class _DotDict:
    """A dictionary that supports dot notation access"""
    def __init__(self, d):
        if d is None:
            d = {}
        for key, value in d.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def keys(self):
        return vars(self).keys()
    
    def items(self):
        return vars(self).items()
    
    def values(self):
        return vars(self).values()


class ReachSet:
    """
    Class for storing reachable sets
    
    This class stores the results of reachability analysis including
    time-point and time-interval reachable sets.
    
    Properties:
        timePoint: Object containing time-point reachable sets
                  - set: List of sets at time points
                  - time: List of time values
                  - error: List of error values (optional)
        timeInterval: Object containing time-interval reachable sets
                     - set: List of sets over time intervals
                     - time: List of time intervals
                     - error: List of error values (optional)
        parent: Index of parent reachable set (default: 0)
        loc: Index of location for hybrid systems (default: 0)
    """
    
    def __init__(self, timePoint: Optional[Dict] = None, timeInterval: Optional[Dict] = None, 
                 parent: int = 0, loc: int = 0):
        """
        Constructor for reachSet
        
        Args:
            timePoint: Time-point reachable sets
            timeInterval: Time-interval reachable sets
            parent: Index of parent reachable set (default: 0)
            loc: Index of location for hybrid systems (default: 0)
        """
        # Initialize empty instantiation
        if timePoint is None and timeInterval is None:
            self.timePoint = _DotDict({})
            self.timeInterval = _DotDict({})
            self.parent = 0
            self.loc = 0
            return
        
        # Validate inputs
        if timePoint is not None and not isinstance(timePoint, dict):
            raise ValueError("timePoint must be a dictionary")
        if timeInterval is not None and not isinstance(timeInterval, dict):
            raise ValueError("timeInterval must be a dictionary")
        if not isinstance(parent, int) or parent < 0:
            raise ValueError("parent must be a non-negative integer")
        if not isinstance(loc, int) or loc < 0:
            raise ValueError("loc must be a non-negative integer")
        
        self.timePoint = _DotDict(timePoint if timePoint is not None else {})
        self.timeInterval = _DotDict(timeInterval if timeInterval is not None else {})
        self.parent = parent
        self.loc = loc
    

    
    @property
    def R0(self):
        """Get initial set (equivalent to MATLAB's R.R0 property)"""
        from cora_python.contSet.emptySet import EmptySet
        from cora_python.g.classes.initialSet import InitialSet

        
        if (not hasattr(self.timePoint, 'set') or 
            not self.timePoint.set or 
            len(self.timePoint.set) == 0):
            # Return empty set for plotting compatibility
            return InitialSet(EmptySet(3))
        else:
            # Return the first time-point set wrapped in InitialSet 
            return InitialSet(self.timePoint.set[0])
    
    
    # String representation
    def __str__(self) -> str:
        """String representation of reachSet"""
        tp_sets = len(self.timePoint.get('set', []))
        ti_sets = len(self.timeInterval.get('set', []))
        return f"reachSet (time-point sets: {tp_sets}, time-interval sets: {ti_sets})"
    
    def __repr__(self) -> str:
        """Detailed string representation of reachSet"""
        return self.__str__()
    
    def __len__(self) -> int:
        """Return the number of branches in the reachSet"""
        # For now, we consider a single reachSet as having length 1
        # In MATLAB, this would be the number of branches/elements in an array of reachSets
        return 1 