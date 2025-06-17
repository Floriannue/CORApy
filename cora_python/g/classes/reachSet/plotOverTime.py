"""
plotOverTime - plots the reachable set over time

This function plots the reachable set over time with various options for
unification and different set types.

Authors: Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Any, List, Union, Tuple
import warnings


def plotOverTime(R, dims: int = 1, unify: bool = False, whichset: str = 'ti', 
                *args, **kwargs) -> Optional[object]:
    """
    Plot the reachable set over time
    
    Args:
        R: ReachSet object or list of ReachSet objects
        dims: Dimension for projection (default: 1)
        unify: Whether to compute union of all reachable sets (default: False)
        whichset: Which set to plot ('ti', 'tp', 'y') (default: 'ti')
        *args: Additional plot arguments
        **kwargs: Additional plot keyword arguments
        
    Returns:
        Handle to the graphics object (or None if no output requested)
    """
    # Parse input arguments
    R, dims, unify, whichset = _parse_input(R, dims, unify, whichset)
    
    # Plot reachable sets
    han = _plot_reach_set(R, dims, unify, whichset, *args, **kwargs)
    
    return han


def _parse_input(R, dims: int, unify: bool, whichset: str) -> Tuple:
    """Parse and validate input arguments"""
    # Ensure R is a list
    if not isinstance(R, list):
        R = [R]
    
    # Validate dimensions
    if not isinstance(dims, int) or dims < 1:
        raise ValueError("dims must be a positive integer")
    
    # Check which set has to be plotted
    whichset = _check_set(R, whichset)
    
    return R, dims, unify, whichset


def _check_set(R: List, whichset: str) -> str:
    """Check and validate which set type to plot"""
    if not whichset:
        whichset = ''
    
    if whichset == 'ti':
        if not R[0].timeInterval or not R[0].timeInterval.get('set'):
            warnings.warn("No time-interval reachable set. Time-point reachable set plotted instead.")
            whichset = 'tp'
    elif whichset == 'tp':
        # No issues (should always be computed)
        pass
    elif whichset == 'y':
        if (not R[0].timeInterval or 
            not R[0].timeInterval.get('algebraic')):
            raise ValueError("No algebraic sets available")
    else:
        # Default value
        if not whichset:
            whichset = 'ti'
            if not R[0].timeInterval or not R[0].timeInterval.get('set'):
                if R[0].timePoint and R[0].timePoint.get('set'):
                    whichset = 'tp'
                else:
                    raise ValueError("No reachable sets available")
        else:
            raise ValueError("whichset must be 'ti', 'tp' or 'y'")
    
    return whichset


def _plot_reach_set(R: List, dims: int, unify: bool, whichset: str, 
                   *args, **kwargs) -> object:
    """Plot reachable set based on options"""
    if unify:
        if whichset in ['ti', 'y']:
            # Check whether fast unify approach is possible
            if _fast_unify_possible(R):
                return _plot_fast_unify(R, dims, whichset, *args, **kwargs)
            else:
                return _plot_unify(R, dims, whichset, *args, **kwargs)
        elif whichset == 'tp':
            return _plot_unify_tp(R, dims, whichset, *args, **kwargs)
    else:
        return _plot_standard(R, dims, whichset, *args, **kwargs)


def _fast_unify_possible(R: List) -> bool:
    """Check whether fast unify approach is possible"""
    for reach_set in R:
        if reach_set.timeInterval and reach_set.timeInterval.get('time'):
            time_intervals = reach_set.timeInterval['time']
            if len(time_intervals) > 1:
                # Check if intervals are disjoint
                for i in range(len(time_intervals) - 1):
                    curr_int = time_intervals[i]
                    next_int = time_intervals[i + 1]
                    
                    # Handle different time interval formats
                    if hasattr(curr_int, 'supremum') and hasattr(next_int, 'infimum'):
                        if curr_int.supremum != next_int.infimum:
                            return False
                    elif isinstance(curr_int, (list, tuple)) and isinstance(next_int, (list, tuple)):
                        if curr_int[1] != next_int[0]:
                            return False
    return True


def _get_x_min_max(set_obj, dims: int) -> Tuple[float, float]:
    """Get minimum and maximum values for a set in given dimension"""
    if hasattr(set_obj, 'c') and hasattr(set_obj, 'G'):
        # Zonotope case - faster conversion
        center = float(set_obj.c[dims - 1, 0]) if dims <= set_obj.c.shape[0] else float(set_obj.c[0, 0])
        if hasattr(set_obj.G, 'shape') and len(set_obj.G.shape) > 1:
            generators = set_obj.G[dims - 1, :] if dims <= set_obj.G.shape[0] else set_obj.G[0, :]
            radius = float(np.sum(np.abs(generators)))
        else:
            radius = 0.0
        return float(center - radius), float(center + radius)
    elif hasattr(set_obj, 'project'):
        # General set with project method
        projected = set_obj.project([dims - 1])  # Convert to 0-based indexing
        if hasattr(projected, 'infimum') and hasattr(projected, 'supremum'):
            return projected.infimum(), projected.supremum()
        elif hasattr(projected, 'interval'):
            interval_proj = projected.interval()
            return interval_proj.infimum(), interval_proj.supremum()
    elif isinstance(set_obj, np.ndarray):
        # Numpy array case
        if dims <= len(set_obj):
            val = float(set_obj[dims - 1])
            return val, val
        else:
            val = float(set_obj[0])
            return val, val
    
    # Fallback
    return 0.0, 0.0


def _plot_fast_unify(R: List, dims: int, whichset: str, *args, **kwargs) -> object:
    """Plot using fast unify algorithm"""
    x_list = []
    y_list = []
    
    # Loop over all reachable sets
    for reach_set in R:
        # Get desired set
        if whichset == 'ti':
            Rset = reach_set.timeInterval.get('set', [])
            Rtime = reach_set.timeInterval.get('time', [])
        elif whichset == 'tp':
            Rset = reach_set.timePoint.get('set', [])
            Rtime = reach_set.timePoint.get('time', [])
        elif whichset == 'y':
            Rset = reach_set.timeInterval.get('algebraic', [])
            Rtime = reach_set.timeInterval.get('time', [])
        
        x_coords = []
        y_coords = []
        
        for j, (set_obj, time_obj) in enumerate(zip(Rset, Rtime)):
            # Get intervals
            int_x_min, int_x_max = _get_x_min_max(set_obj, dims)
            
            # Get time interval bounds
            if hasattr(time_obj, 'infimum') and hasattr(time_obj, 'supremum'):
                t_min, t_max = time_obj.infimum(), time_obj.supremum()
            elif isinstance(time_obj, (list, tuple)) and len(time_obj) == 2:
                t_min, t_max = time_obj[0], time_obj[1]
            else:
                t_min = t_max = float(time_obj)
            
            # Add coordinates for polygon (clockwise)
            x_coords.extend([t_min, t_max])
            y_coords.extend([int_x_max, int_x_max])
            
            # Add reverse coordinates for bottom
            x_coords.insert(0, t_min)
            x_coords.insert(0, t_max)
            y_coords.insert(0, int_x_min)
            y_coords.insert(0, int_x_min)
        
        x_list.extend(x_coords)
        y_list.extend(y_coords)
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot as filled polygon
    if x_list and y_list:
        ax.fill(x_list, y_list, alpha=0.5, *args, **kwargs)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(f'State Dimension {dims}')
    ax.set_title('Reachable Set Over Time')
    ax.grid(True, alpha=0.3)
    
    return fig


def _plot_unify(R: List, dims: int, whichset: str, *args, **kwargs) -> object:
    """Plot using polygon unification"""
    # This is a simplified version - full implementation would require
    # polygon union operations
    return _plot_standard(R, dims, whichset, *args, **kwargs)


def _plot_unify_tp(R: List, dims: int, whichset: str, *args, **kwargs) -> object:
    """Plot time-point solutions with NaN breaks"""
    t_all = []
    x_all = []
    
    # Loop over all reachable sets
    for reach_set in R:
        Rset = reach_set.timePoint.get('set', [])
        Rtime = reach_set.timePoint.get('time', [])
        
        for set_obj, time_obj in zip(Rset, Rtime):
            # Get bounds for set
            int_x_min, int_x_max = _get_x_min_max(set_obj, dims)
            
            # Get time value
            t_val = float(time_obj)
            
            # Add time points (repeated for min/max)
            t_all.extend([t_val, t_val, np.nan])
            x_all.extend([int_x_min, int_x_max, np.nan])
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if t_all and x_all:
        ax.plot(t_all, x_all, *args, **kwargs)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(f'State Dimension {dims}')
    ax.set_title('Reachable Set Over Time (Time Points)')
    ax.grid(True, alpha=0.3)
    
    return fig


def _plot_standard(R: List, dims: int, whichset: str, *args, **kwargs) -> object:
    """Plot reachable sets as individual rectangles"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loop over all reachable sets
    for i, reach_set in enumerate(R):
        # Get desired set
        if whichset == 'ti':
            if reach_set.timeInterval and reach_set.timeInterval.get('set'):
                Rset = reach_set.timeInterval['set']
                Rtime = reach_set.timeInterval['time']
            elif reach_set.timePoint and reach_set.timePoint.get('set'):
                Rset = reach_set.timePoint['set']
                Rtime = reach_set.timePoint['time']
            else:
                continue
        elif whichset == 'tp':
            if not reach_set.timePoint or not reach_set.timePoint.get('set'):
                continue
            Rset = reach_set.timePoint['set']
            Rtime = reach_set.timePoint['time']
        elif whichset == 'y':
            if (not reach_set.timeInterval or 
                not reach_set.timeInterval.get('algebraic')):
                continue
            Rset = reach_set.timeInterval['algebraic']
            Rtime = reach_set.timeInterval['time']
        
        # Plot each set as a rectangle
        for set_obj, time_obj in zip(Rset, Rtime):
            # Get bounds
            int_x_min, int_x_max = _get_x_min_max(set_obj, dims)
            
            # Get time bounds
            if hasattr(time_obj, 'infimum') and hasattr(time_obj, 'supremum'):
                t_min, t_max = time_obj.infimum(), time_obj.supremum()
            elif isinstance(time_obj, (list, tuple)) and len(time_obj) == 2:
                t_min, t_max = time_obj[0], time_obj[1]
            else:
                t_min = t_max = float(time_obj)
            
            # Create rectangle
            width = float(t_max - t_min) if t_max != t_min else 0.01  # Small width for time points
            height = float(int_x_max - int_x_min)
            
            rect = plt.Rectangle((float(t_min), float(int_x_min)), width, height, 
                               alpha=0.5, *args, **kwargs)
            ax.add_patch(rect)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(f'State Dimension {dims}')
    ax.set_title('Reachable Set Over Time')
    ax.grid(True, alpha=0.3)
    ax.autoscale()
    
    return fig 