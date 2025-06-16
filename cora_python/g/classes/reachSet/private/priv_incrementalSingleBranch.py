"""
priv_incrementalSingleBranch - incremental STL verification using a reachable set with a single branch

This function performs incremental STL verification for reachable sets with
a single branch.

Authors: Florian Lercher (MATLAB)
         Python translation by AI Assistant
Written: 15-February-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import List, Tuple, Any, Union, Optional
import warnings


def priv_incrementalSingleBranch(R, analyzer, verbose: bool = False) -> 'FourValuedResult':
    """
    Incremental STL verification using a reachable set with a single branch
    
    Args:
        R: ReachSet object with just one branch
        analyzer: OnlineReachSetAnalyzer object
        verbose: Whether to print additional information
        
    Returns:
        FourValuedResult: Four-valued verdict for the current branch
    """
    # This is a simplified implementation
    warnings.warn("priv_incrementalSingleBranch is simplified - full framework needed")
    
    # Ensure that the reachable set has only one branch
    if isinstance(R, list) and len(R) > 1:
        raise ValueError("Expected reachable set with a single branch")
    
    # Handle single ReachSet object
    if isinstance(R, list):
        R = R[0]
    
    # Process time interval solutions
    if R.timeInterval and R.timeInterval.get('time') and R.timeInterval.get('set'):
        time_intervals = R.timeInterval['time']
        sets = R.timeInterval['set']
        
        for j, (time_int, reach_set) in enumerate(zip(time_intervals, sets)):
            # Observe the time interval solution
            t = time_int
            set_obj = reach_set
            
            # Create STL interval from t
            if hasattr(t, 'infimum') and hasattr(t, 'supremum'):
                lb = float(t.infimum())
                ub = float(t.supremum())
            elif isinstance(t, (list, tuple)) and len(t) == 2:
                lb, ub = float(t[0]), float(t[1])
            else:
                lb = ub = float(t)
            
            # If we are not at the end, excluding the upper time bound is fine,
            # because the time point solution is included in the next interval solution anyway
            rc = j == len(time_intervals) - 1
            stl_int = _stl_interval(lb, ub, True, rc)
            
            # Observe the reachable set
            analyzer.observe_set(set_obj, stl_int, R.loc)
            
            # Obtain the current verdict and check if it is conclusive
            v = analyzer.get_verdict()
            if not _is_inconclusive(v):
                if verbose:
                    print(f'Stop reachability analysis early at time {ub}')
                return v
    
    # Make sure that all observations are propagated
    analyzer.force_propagation()
    
    return analyzer.get_verdict()


def _stl_interval(lb: float, ub: float, left_closed: bool = True, 
                 right_closed: bool = True) -> 'STLInterval':
    """Create STL interval (simplified)"""
    class STLInterval:
        def __init__(self, lb, ub, left_closed, right_closed):
            self.lb = lb
            self.ub = ub
            self.left_closed = left_closed
            self.right_closed = right_closed
    
    return STLInterval(lb, ub, left_closed, right_closed)


def _is_inconclusive(verdict) -> bool:
    """Check if verdict is inconclusive (simplified)"""
    if hasattr(verdict, 'value'):
        return verdict.value == 'Inconclusive'
    return False 