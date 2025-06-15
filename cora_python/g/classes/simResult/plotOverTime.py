"""
plotOverTime - plots the simulated trajectories over time

Syntax:
    han = plotOverTime(simRes)
    han = plotOverTime(simRes, dims)
    han = plotOverTime(simRes, dims, **kwargs)

Inputs:
    simRes - simResult object
    dims - (optional) dimensions for projection (default: [0])
    **kwargs - (optional) plot settings including:
        'Traj' - which trajectory to plot ('x', 'y', 'a')
        Other matplotlib plotting options

Outputs:
    han - handle to the graphics object

Authors: Niklas Kochdumper, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 06-June-2020 (MATLAB)
Last update: 22-March-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Any, Union


def plotOverTime(simRes, dims: Optional[Union[int, List[int]]] = None, **kwargs) -> Any:
    """
    Plot the simulated trajectories over time
    
    Args:
        simRes: simResult object
        dims: Dimensions for projection (default: [0])
        **kwargs: Plot settings
        
    Returns:
        Handle to the graphics object
    """
    # Set default dimensions
    if dims is None:
        dims = [0]
    elif isinstance(dims, int):
        dims = [dims]
    
    # Parse options
    whichtraj = kwargs.pop('Traj', 'x')  # 'x', 'y', or 'a'
    
    # Check if simResult is empty
    from .isemptyobject import isemptyobject
    if isemptyobject(simRes):
        # Plot empty trajectory
        return plt.plot([], [], **kwargs)
    
    # Determine which trajectory to plot
    if whichtraj == 'x':
        trajs_to_plot = simRes.x
        times_to_plot = simRes.t
    elif whichtraj == 'y':
        if not simRes.y:
            raise ValueError("No output trajectories available")
        trajs_to_plot = simRes.y
        times_to_plot = simRes.t
    elif whichtraj == 'a':
        if not simRes.a:
            raise ValueError("No algebraic trajectories available")
        trajs_to_plot = simRes.a
        times_to_plot = simRes.t
    else:
        raise ValueError(f"Unknown trajectory type: {whichtraj}")
    
    if not trajs_to_plot or not times_to_plot:
        # Plot empty trajectory
        return plt.plot([], [], **kwargs)
    
    # Plot the trajectories over time
    handles = []
    
    for i, (traj, time) in enumerate(zip(trajs_to_plot, times_to_plot)):
        if traj.size == 0 or time.size == 0:
            continue
        
        # Extract the desired dimensions
        if traj.ndim == 1:
            # Single point trajectory
            traj_proj = traj[dims] if len(dims) <= len(traj) else traj
            if len(dims) == 1:
                traj_proj = [traj_proj]  # Make it a list for consistent handling
        else:
            # Multi-point trajectory
            if max(dims) < traj.shape[1]:
                traj_proj = traj[:, dims]
            else:
                # Handle case where requested dimensions exceed available dimensions
                available_dims = [d for d in dims if d < traj.shape[1]]
                if available_dims:
                    traj_proj = traj[:, available_dims]
                else:
                    continue  # Skip this trajectory
        
        # Ensure time is 1D
        if time.ndim > 1:
            time = time.flatten()
        
        # Plot each dimension
        if traj.ndim == 1:
            # Single time point
            for j, dim_idx in enumerate(dims):
                if dim_idx < len(traj):
                    handle = plt.plot(time, [traj[dim_idx]], 'o', **kwargs)
                    handles.append(handle)
        else:
            # Multiple time points
            if len(dims) == 1:
                # Single dimension
                handle = plt.plot(time, traj_proj, **kwargs)
                handles.append(handle)
            else:
                # Multiple dimensions - plot each separately
                for j, dim_idx in enumerate(dims):
                    if dim_idx < traj.shape[1]:
                        handle = plt.plot(time, traj[:, dim_idx], **kwargs)
                        handles.append(handle)
        
        # For subsequent plots, don't show in legend
        if i == 0:
            kwargs['label'] = kwargs.get('label', '_nolegend_')
    
    # Set labels
    plt.xlabel('Time')
    if len(dims) == 1:
        plt.ylabel(f'{whichtraj}_{dims[0]}')
    else:
        plt.ylabel(f'{whichtraj}')
    
    # Return the first handle or all handles
    if len(handles) == 1:
        return handles[0]
    elif len(handles) > 1:
        return handles
    else:
        return None 