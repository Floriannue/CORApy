"""
plotTimeStep - plots the time step size used in reachSet object over time

This function plots the time step size used in reachSet object over time
(all in one graph).

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 18-June-2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Any, List, Union


def plotTimeStep(R, *args, **kwargs) -> Optional[object]:
    """
    Plot the time step size used in reachSet object over time
    
    Args:
        R: ReachSet object or list of ReachSet objects
        *args: Additional plot arguments (LineSpec)
        **kwargs: Additional plot keyword arguments (Name-Value pairs)
        
    Returns:
        Handle to the graphics object (or None if no output requested)
    """
    # Ensure R is a list
    if not isinstance(R, list):
        R = [R]
    
    # Initialize min/max values for axis limits
    min_timestep = float('inf')
    max_timestep = float('-inf')
    cumsum_min = float('inf')
    cumsum_max = float('-inf')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loop over all branches in R
    for i, reach_set in enumerate(R):
        if reach_set.timePoint and reach_set.timePoint.get('time'):
            # Time axis - use time points
            time_points = reach_set.timePoint['time']
            
            # Convert to numpy array if needed
            if isinstance(time_points, list):
                time_array = np.array([t if np.isscalar(t) else t[0] for t in time_points])
            else:
                time_array = np.array(time_points)
            
            # Calculate time step sizes
            time_steps = np.diff(np.concatenate([[0], time_array]))
            
            # Create cumulative time vector for plotting
            cumsum_t_vec = np.concatenate([[0], 
                                         np.repeat(time_array[:-1], 2), 
                                         [time_array[-1]]])
            
            # Create time step vector for plotting (repeated for step plot)
            t_vec = np.repeat(time_steps, 2)
            
        elif reach_set.timeInterval and reach_set.timeInterval.get('time'):
            # Use time intervals
            time_intervals = reach_set.timeInterval['time']
            t_vec = []
            
            for j, time_int in enumerate(time_intervals):
                if hasattr(time_int, '__len__') and len(time_int) == 2:
                    # Time interval [start, end]
                    step_size = time_int[1] - time_int[0]
                elif hasattr(time_int, 'rad'):
                    # Interval object with rad method
                    step_size = 2 * time_int.rad()
                else:
                    # Assume scalar time step
                    step_size = float(time_int)
                
                t_vec.append(step_size)
            
            t_vec = np.array(t_vec)
            
            # Create cumulative time vector
            cumsum_t_vec = np.concatenate([[0], 
                                         np.repeat(np.cumsum(t_vec[:-1]), 2), 
                                         [np.sum(t_vec)]])
            
            # Repeat for step plot
            t_vec = np.repeat(t_vec, 2)
        
        else:
            # No time information available
            continue
        
        # Plot the time step sizes
        ax.plot(cumsum_t_vec, t_vec, *args, **kwargs, label=f'Branch {i+1}')
        
        # Update axis limits
        if len(t_vec) > 0:
            min_timestep = min(min_timestep, np.min(t_vec))
            max_timestep = max(max_timestep, np.max(t_vec))
            cumsum_min = min(cumsum_min, cumsum_t_vec[0])
            cumsum_max = max(cumsum_max, cumsum_t_vec[-1])
    
    # Set title and labels
    ax.set_title('ReachSet: Time Step Size')
    ax.set_xlabel('t')
    ax.set_ylabel('Time Step Size')
    
    # Set axis limits if we have valid data
    if min_timestep != float('inf') and max_timestep != float('-inf'):
        ax.set_xlim([cumsum_min, cumsum_max])
        ax.set_ylim([0.9 * min_timestep, 1.1 * max_timestep])
    
    # Add grid and legend if multiple branches
    ax.grid(True, alpha=0.3)
    if len(R) > 1:
        ax.legend()
    
    plt.tight_layout()
    
    return fig 