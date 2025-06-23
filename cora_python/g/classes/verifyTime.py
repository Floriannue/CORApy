"""
verifyTime - 1D interval representing the times at which a specification
must be checked (verified/falsified); crucially, the interval may
consist of multiple disjoint partial intervals

Syntax:
    Itime = VerifyTime(bounds)

Inputs:
    bounds - array of time intervals (optional)
    
Outputs:
    Itime - generated VerifyTime object

Example:
    Itime = VerifyTime([[0,2], [4,5]])

References:
    [1] M. Wetzlinger et al. "Fully automated verification of linear
        systems using inner-and outer-approximations of reachable sets",
        TAC, 2023.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: LinearSys/private/priv_reach_adaptive

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 10-November-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class VerifyTime:
    """
    1D interval representing the times at which a specification must be checked
    """
    
    def __init__(self, bounds: Optional[Union[List[List[float]], np.ndarray]] = None):
        """
        Constructor for VerifyTime
        
        Args:
            bounds: Array of time intervals, each row [start, end]
        """
        if bounds is None:
            self.bounds = np.zeros((0, 2))
        elif isinstance(bounds, VerifyTime):
            # Copy constructor
            self.bounds = bounds.bounds.copy()
        else:
            bounds = np.array(bounds)
            if bounds.ndim == 1 and len(bounds) == 2:
                bounds = bounds.reshape(1, -1)
            
            self._checkInputArgs(bounds)
            self.bounds = bounds
    
    @property
    def intervals(self) -> List[List[float]]:
        """
        Get time intervals as list of [start, end] pairs (for compatibility with tests)
        
        Returns:
            List of time intervals
        """
        if self.bounds.size == 0:
            return []
        return self.bounds.tolist()
    
    @intervals.setter
    def intervals(self, value: List[List[float]]):
        """
        Set time intervals from list of [start, end] pairs
        
        Args:
            value: List of time intervals
        """
        if not value:
            self.bounds = np.zeros((0, 2))
        else:
            bounds = np.array(value)
            if bounds.ndim == 1 and len(bounds) == 2:
                bounds = bounds.reshape(1, -1)
            self._checkInputArgs(bounds)
            self.bounds = bounds
    
    def _checkInputArgs(self, bounds: np.ndarray):
        """
        Check correctness of input arguments
        
        Args:
            bounds: Array of time intervals to check
        """
        if bounds.size == 0:
            return
        
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise CORAerror('CORA:wrongInputInConstructor',
                           'Time intervals must be specified as nx2 array.')
        
        # Check for valid intervals
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise CORAerror('CORA:wrongInputInConstructor',
                           'Start time must be less than or equal to end time.')
        
        # Check for non-negative times
        if np.any(bounds < 0):
            raise CORAerror('CORA:wrongInputInConstructor',
                           'Time values must be non-negative.')
        
        # Check for finite values
        if np.any(~np.isfinite(bounds)):
            raise CORAerror('CORA:wrongInputInConstructor',
                           'Time values must be finite.')
        
        # Check for proper ordering (non-overlapping intervals) - matching MATLAB behavior
        if bounds.shape[0] > 1:
            for i in range(bounds.shape[0] - 1):
                if bounds[i, 1] > bounds[i + 1, 0]:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                   'Time intervals must be non-overlapping and ordered.')
    
    def shift(self, t: float) -> 'VerifyTime':
        """
        Shift time intervals by t
        
        Args:
            t: Time shift amount
            
        Returns:
            Shifted VerifyTime object
        """
        result = VerifyTime()
        result.bounds = self.bounds + t
        return result
    
    def startTime(self) -> float:
        """
        Read out minimum start time over all intervals
        
        Returns:
            Minimum start time
        """
        if self.bounds.size == 0:
            return 0.0
        return np.min(self.bounds[:, 0])
    
    def finalTime(self) -> float:
        """
        Read out maximum final time over all intervals
        
        Returns:
            Maximum final time
        """
        if self.bounds.size == 0:
            return 0.0
        return np.max(self.bounds[:, 1])
    
    def numIntervals(self) -> int:
        """
        Read out number of disjoint time intervals
        
        Returns:
            Number of intervals
        """
        return self.bounds.shape[0]
    
    def compact(self, tol: Optional[float] = None) -> 'VerifyTime':
        """
        Unify adjacent intervals (if possible)
        
        Args:
            tol: Tolerance for merging intervals
            
        Returns:
            Compacted VerifyTime object
        """
        # Quick exit: already in minimal form
        if self.numIntervals() <= 1:
            return VerifyTime(self.bounds.copy())
        
        if tol is None:
            tol = np.finfo(float).eps * (self.finalTime() - self.startTime())
        
        bounds = self.bounds.copy()
        
        # Find indices where the end matches the next start
        idx = np.abs(bounds[:-1, 1] - bounds[1:, 0]) <= tol
        
        # Quick exits: all match/no matches
        if np.all(idx):
            return VerifyTime([[self.startTime(), self.finalTime()]])
        elif not np.any(idx):
            return VerifyTime(bounds)
        
        # Merge adjacent intervals
        result_bounds = []
        i = 0
        while i < len(bounds):
            start = bounds[i, 0]
            end = bounds[i, 1]
            
            # Find consecutive intervals to merge
            while i < len(bounds) - 1 and idx[i]:
                i += 1
                end = bounds[i, 1]
            
            result_bounds.append([start, end])
            i += 1
        
        return VerifyTime(result_bounds)
    
    def representsa(self, type_str: str, tol: float = 0.0) -> bool:
        """
        Check if VerifyTime object represents a specific type
        
        Args:
            type_str: Type to check ('emptySet', 'interval')
            tol: Tolerance for comparison
            
        Returns:
            True if represents the specified type
        """
        if type_str == 'emptySet':
            return self.bounds.shape[0] == 0
        elif type_str == 'interval':
            compacted = self.compact(tol)
            return compacted.bounds.shape[0] == 1
        else:
            raise CORAerror('CORA:notSupported',
                           'Only comparison to emptySet and interval supported.')
    
    def isequal(self, other: 'VerifyTime') -> bool:
        """
        Check equality with another VerifyTime object
        
        Args:
            other: Other VerifyTime object
            
        Returns:
            True if equal
        """
        if not isinstance(other, VerifyTime):
            raise CORAerror('CORA:wrongInputInConstructor',
                           'Comparison only implemented for VerifyTime objects.')
        
        # Case where one or both are empty
        self_empty = self.representsa('emptySet')
        other_empty = other.representsa('emptySet')
        
        if self_empty != other_empty:
            return False
        elif self_empty and other_empty:
            return True
        
        # Rewrite both in minimal representation
        self_compact = self.compact()
        other_compact = other.compact()
        
        if self_compact.bounds.shape != other_compact.bounds.shape:
            return False
        
        return np.allclose(self_compact.bounds, other_compact.bounds)
    
    def __eq__(self, other) -> bool:
        """
        Equality operator
        
        Args:
            other: Other object to compare
            
        Returns:
            True if equal
        """
        if not isinstance(other, VerifyTime):
            return False
        return self.isequal(other)
    
    def contains(self, t: float) -> bool:
        """
        Check if time t is contained in any interval
        
        Args:
            t: Time to check
            
        Returns:
            True if t is contained
        """
        if self.bounds.size == 0:
            return False
        
        return np.any((self.bounds[:, 0] <= t) & (t <= self.bounds[:, 1]))
    
    def timeUntilSwitch(self, t: float) -> Tuple[float, bool]:
        """
        Compute time until next switch and whether full computation is needed
        
        Args:
            t: Current time
            
        Returns:
            Tuple of (time until switch, full computation needed)
        """
        if self.bounds.size == 0:
            return float('inf'), False
        
        # Check if current time is inside any interval
        inside_interval = self.contains(t)
        
        if inside_interval:
            # Find the end of the current interval
            current_intervals = self.bounds[(self.bounds[:, 0] <= t) & (t <= self.bounds[:, 1])]
            if len(current_intervals) > 0:
                # Time until current interval ends
                time_until_switch = np.min(current_intervals[:, 1]) - t
                return time_until_switch, True
        
        # Not inside any interval - find next interval start
        future_starts = self.bounds[:, 0][self.bounds[:, 0] > t]
        if len(future_starts) > 0:
            time_until_switch = np.min(future_starts) - t
            return time_until_switch, False
        
        # No future intervals
        return float('inf'), False
    
    def remove(self, t_start: float, t_end: float) -> 'VerifyTime':
        """
        Remove time interval from VerifyTime object
        
        Args:
            t_start: Start time to remove
            t_end: End time to remove
            
        Returns:
            VerifyTime object with interval removed
        """
        if self.bounds.size == 0:
            return VerifyTime()
        
        new_bounds = []
        
        for i in range(self.bounds.shape[0]):
            interval_start, interval_end = self.bounds[i]
            
            # No overlap
            if interval_end < t_start or interval_start > t_end:
                new_bounds.append([interval_start, interval_end])
            # Partial overlap - keep non-overlapping parts
            elif interval_start < t_start and interval_end > t_end:
                # Split interval
                new_bounds.append([interval_start, t_start])
                new_bounds.append([t_end, interval_end])
            elif interval_start < t_start and interval_end <= t_end:
                # Keep left part
                if interval_start < t_start:
                    new_bounds.append([interval_start, t_start])
            elif interval_start >= t_start and interval_end > t_end:
                # Keep right part
                if interval_end > t_end:
                    new_bounds.append([t_end, interval_end])
            # Complete overlap - remove entire interval (don't add to new_bounds)
        
        return VerifyTime(new_bounds) if new_bounds else VerifyTime() 