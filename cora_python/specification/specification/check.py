"""
check - checks if a set satisfies the specification

This function verifies whether a given set, trajectory, or point cloud
satisfies a specification or list of specifications.

Syntax:
    res, indSpec, indObj = check(spec, S)
    res, indSpec, indObj = check(spec, S, time)

Inputs:
    spec - specification object or list of specifications
    S - numeric array, contSet, reachSet, or simResult object
    time - (optional) time values corresponding to S

Outputs:
    res - True/False whether set satisfies all specifications
    indSpec - index of the first specification that is violated (None if all pass)
    indObj - index of the first object S that is violated (None if all pass)

Example:
    # Check if point satisfies safety specification
    safe_set = Interval([-5, -5], [5, 5])
    spec = Specification(safe_set, 'safeSet')
    point = np.array([[1], [2]])
    res, indSpec, indObj = check(spec, point)

Authors: Niklas Kochdumper, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: 24-May-2024 (TL, vectorized check for numeric input) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Any
from .specification import Specification


def check(spec: Union[Specification, List[Specification]], 
          S: Union[np.ndarray, Any], 
          time: Optional[Union[float, np.ndarray]] = None) -> Tuple[bool, Optional[int], Optional[Union[int, tuple]]]:
    """
    Check if a set satisfies the specification(s)
    
    Args:
        spec: Specification object or list of specifications
        S: Set, trajectory, or points to check
        time: Optional time information
        
    Returns:
        Tuple of (result, spec_index, object_index)
    """
    
    # Convert single specification to list
    if isinstance(spec, Specification):
        spec_list = [spec]
    else:
        spec_list = spec
    
    # Handle numeric input (point cloud)
    if isinstance(S, np.ndarray):
        return _check_numeric(spec_list, S, time)
    
    # Handle contSet objects
    elif hasattr(S, '__class__') and hasattr(S, 'contains'):
        return _check_contSet(spec_list, S, time)
    
    # Handle other types (reachSet, simResult) - simplified implementation
    else:
        # For now, assume it's a single set and try to check it
        for i, spec_obj in enumerate(spec_list):
            if not spec_obj.check(S, time):
                return False, i, 0
        return True, None, None


def _check_numeric(spec_list: List[Specification], 
                   S: np.ndarray, 
                   time: Optional[Union[float, np.ndarray]]) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Check numeric input (point cloud) against specifications
    
    Args:
        spec_list: List of specifications
        S: Point cloud (n x num_points)
        time: Time values (scalar or array)
        
    Returns:
        Tuple of (result, spec_index, point_index)
    """
    
    # Ensure S is 2D
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    num_points = S.shape[1]
    
    # Handle time input
    if time is not None:
        if np.isscalar(time):
            time_array = np.full(num_points, time)
        else:
            time_array = np.asarray(time)
            if time_array.size != num_points:
                raise ValueError("Time array size must match number of points")
    else:
        time_array = None
    
    # Check each specification
    for spec_idx, spec_obj in enumerate(spec_list):
        
        # Check each point
        for point_idx in range(num_points):
            point = S[:, point_idx:point_idx+1]
            point_time = time_array[point_idx] if time_array is not None else None
            
            # Check if specification is active at this time
            if not spec_obj.is_active(point_time) if point_time is not None else True:
                continue
            
            # Check the specification
            if not _check_single_point(spec_obj, point, point_time):
                return False, spec_idx, point_idx
    
    return True, None, None


def _check_single_point(spec_obj: Specification, 
                       point: np.ndarray, 
                       time: Optional[float]) -> bool:
    """
    Check a single point against a specification
    
    Args:
        spec_obj: Specification object
        point: Single point (n x 1)
        time: Time value
        
    Returns:
        bool: True if point satisfies specification
    """
    
    try:
        if spec_obj.type == 'safeSet':
            # Point must be in safe set
            return spec_obj.set.contains(point.flatten())
        
        elif spec_obj.type == 'unsafeSet':
            # Point must not be in unsafe set
            return not spec_obj.set.contains(point.flatten())
        
        elif spec_obj.type == 'invariant':
            # Point must be in invariant set
            return spec_obj.set.contains(point.flatten())
        
        else:
            # Unknown type - assume violation
            return False
            
    except Exception:
        # If contains check fails, assume violation
        return False


def _check_contSet(spec_list: List[Specification], 
                   S: Any, 
                   time: Optional[Union[float, np.ndarray]]) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Check contSet object against specifications
    
    Args:
        spec_list: List of specifications
        S: contSet object
        time: Time information
        
    Returns:
        Tuple of (result, spec_index, object_index)
    """
    
    # Check each specification
    for spec_idx, spec_obj in enumerate(spec_list):
        
        # Use the specification's check method
        if not spec_obj.check(S, time):
            return False, spec_idx, 0
    
    return True, None, None 