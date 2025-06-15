"""
plot - plots a projection of the simulated trajectories

Syntax:
    han = plot(simRes)
    han = plot(simRes, dims)
    han = plot(simRes, dims, **kwargs)

Inputs:
    simRes - simResult object
    dims - (optional) dimensions for projection (default: [0, 1])
    **kwargs - (optional) plot settings including:
        'Traj' - which trajectory to plot ('x', 'y', 'a')
        Other matplotlib plotting options

Outputs:
    han - handle to the graphics object

Authors: Niklas Kochdumper, Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 06-June-2020 (MATLAB)
Last update: 28-July-2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Any


def plot(simRes, dims: Optional[List[int]] = None, **kwargs) -> Any:
    """
    Plot a projection of the simulated trajectories
    
    Args:
        simRes: simResult object
        dims: Dimensions for projection (default: [0, 1])
        **kwargs: Plot settings
        
    Returns:
        Handle to the graphics object
    """
    # Set default dimensions
    if dims is None:
        dims = [0, 1]
    
    # Validate inputs
    if len(dims) < 2:
        raise ValueError("At least 2 dimensions required for plotting")
    elif len(dims) > 3:
        raise ValueError("At most 3 dimensions supported for plotting")
    
    # Parse options
    whichtraj = kwargs.pop('Traj', 'x')  # 'x', 'y', or 'a'
    
    # Check if simResult is empty
    from .isemptyobject import isemptyobject
    if isemptyobject(simRes):
        # Plot empty trajectory
        if len(dims) == 2:
            return plt.plot([], [], **kwargs)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            return ax.plot([], [], [], **kwargs)
    
    # Determine which trajectory to plot
    if whichtraj == 'x':
        trajs_to_plot = simRes.x
    elif whichtraj == 'y':
        if not simRes.y:
            raise ValueError("No output trajectories available")
        trajs_to_plot = simRes.y
    elif whichtraj == 'a':
        if not simRes.a:
            raise ValueError("No algebraic trajectories available")
        trajs_to_plot = simRes.a
    else:
        raise ValueError(f"Unknown trajectory type: {whichtraj}")
    
    if not trajs_to_plot:
        # Plot empty trajectory
        if len(dims) == 2:
            return plt.plot([], [], **kwargs)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            return ax.plot([], [], [], **kwargs)
    
    # Plot the trajectories
    handles = []
    
    for i, traj in enumerate(trajs_to_plot):
        if traj.size == 0:
            continue
        
        # Extract the desired dimensions
        if traj.ndim == 1:
            # Single point trajectory
            traj_proj = traj[dims] if len(dims) <= len(traj) else traj
        else:
            # Multi-point trajectory
            traj_proj = traj[:, dims] if traj.shape[1] > max(dims) else traj
        
        # Plot the trajectory
        if len(dims) == 2:
            if traj_proj.ndim == 1:
                handle = plt.plot(traj_proj[0], traj_proj[1], 'o', **kwargs)
            else:
                handle = plt.plot(traj_proj[:, 0], traj_proj[:, 1], **kwargs)
        else:  # 3D
            fig = plt.gcf()
            if not fig.axes or not hasattr(fig.axes[0], 'zaxis'):
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.axes[0]
            
            if traj_proj.ndim == 1:
                handle = ax.plot([traj_proj[0]], [traj_proj[1]], [traj_proj[2]], 'o', **kwargs)
            else:
                handle = ax.plot(traj_proj[:, 0], traj_proj[:, 1], traj_proj[:, 2], **kwargs)
        
        handles.append(handle)
        
        # For subsequent plots, don't show in legend
        if i == 0:
            kwargs['label'] = kwargs.get('label', '_nolegend_')
    
    # Set 3D view if needed
    if len(dims) == 3:
        plt.gca().view_init()
    
    # Return the first handle or all handles
    if len(handles) == 1:
        return handles[0]
    elif len(handles) > 1:
        return handles
    else:
        return None 