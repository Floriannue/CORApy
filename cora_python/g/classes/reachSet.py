"""
reachSet - class for storing reachable sets

This class stores reachable sets computed by reachability analysis algorithms.
It contains both time-point and time-interval reachable sets.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-May-2007 (MATLAB)
Last update: 22-September-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union


class ReachSet:
    """
    Class for storing reachable sets
    
    This class stores the results of reachability analysis including
    time-point and time-interval reachable sets.
    
    Properties:
        timePoint: Dictionary containing time-point reachable sets
                  - set: List of sets at time points
                  - time: List of time values
                  - error: List of error values (optional)
        timeInterval: Dictionary containing time-interval reachable sets
                     - set: List of sets over time intervals
                     - time: List of time intervals
                     - error: List of error values (optional)
    """
    
    def __init__(self, timePoint: Optional[Dict] = None, timeInterval: Optional[Dict] = None):
        """
        Constructor for reachSet
        
        Args:
            timePoint: Time-point reachable sets
            timeInterval: Time-interval reachable sets
        """
        self.timePoint = timePoint if timePoint is not None else {}
        self.timeInterval = timeInterval if timeInterval is not None else {}
    
    @staticmethod
    def initReachSet(timePoint: Dict, timeInterval: Optional[Dict] = None) -> 'ReachSet':
        """
        Initialize a reachSet object
        
        Args:
            timePoint: Time-point reachable sets
            timeInterval: Time-interval reachable sets (optional)
            
        Returns:
            ReachSet object
        """
        return ReachSet(timePoint, timeInterval)
    
    def query(self, property_name: str, time_value: Optional[float] = None) -> Any:
        """
        Query reachable sets for specific properties
        
        Args:
            property_name: Property to query ('reachSet', 'timePoint', 'timeInterval')
            time_value: Specific time value (optional)
            
        Returns:
            Requested property value
        """
        if property_name == 'reachSet':
            return self
        elif property_name == 'timePoint':
            if time_value is not None:
                # Find closest time point
                if 'time' in self.timePoint:
                    times = np.array(self.timePoint['time'])
                    idx = np.argmin(np.abs(times - time_value))
                    return self.timePoint['set'][idx]
            return self.timePoint
        elif property_name == 'timeInterval':
            return self.timeInterval
        else:
            raise ValueError(f"Unknown property: {property_name}")
    
    def contains(self, simResult) -> bool:
        """
        Check if simulation results are contained in reachable sets
        
        Args:
            simResult: Simulation result object
            
        Returns:
            True if all trajectories are contained
        """
        # TODO: Implement containment check
        # This would check if all simulation trajectories are contained
        # within the corresponding reachable sets
        raise NotImplementedError("Containment check not yet implemented")
    
    def project(self, dims: List[int]) -> 'ReachSet':
        """
        Project reachable sets to lower-dimensional subspace
        
        Args:
            dims: Dimensions to project to
            
        Returns:
            Projected reachSet
        """
        projected_timePoint = {}
        projected_timeInterval = {}
        
        # Project time-point sets
        if 'set' in self.timePoint:
            projected_timePoint['set'] = []
            for s in self.timePoint['set']:
                if hasattr(s, 'project'):
                    projected_timePoint['set'].append(s.project(dims))
                else:
                    # For numeric arrays, just select dimensions
                    if isinstance(s, np.ndarray):
                        projected_timePoint['set'].append(s[dims])
                    else:
                        projected_timePoint['set'].append(s)
            
            # Copy other fields
            for key in ['time', 'error']:
                if key in self.timePoint:
                    projected_timePoint[key] = self.timePoint[key].copy()
        
        # Project time-interval sets
        if 'set' in self.timeInterval:
            projected_timeInterval['set'] = []
            for s in self.timeInterval['set']:
                if hasattr(s, 'project'):
                    projected_timeInterval['set'].append(s.project(dims))
                else:
                    # For numeric arrays, just select dimensions
                    if isinstance(s, np.ndarray):
                        projected_timeInterval['set'].append(s[dims])
                    else:
                        projected_timeInterval['set'].append(s)
            
            # Copy other fields
            for key in ['time', 'error']:
                if key in self.timeInterval:
                    projected_timeInterval[key] = self.timeInterval[key].copy()
        
        return ReachSet(projected_timePoint, projected_timeInterval)
    
    def __str__(self) -> str:
        """String representation of reachSet"""
        tp_sets = len(self.timePoint.get('set', []))
        ti_sets = len(self.timeInterval.get('set', []))
        return f"reachSet (time-point sets: {tp_sets}, time-interval sets: {ti_sets})"
    
    def __repr__(self) -> str:
        """Detailed string representation of reachSet"""
        return self.__str__() 