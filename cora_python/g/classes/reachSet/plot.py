"""
plot - plots a projection of the reachable set

Syntax:
    han = plot(R)
    han = plot(R, dims)
    han = plot(R, dims, **kwargs)

Inputs:
    R - reachSet object
    dims - (optional) dimensions for projection (default: [0, 1])
    **kwargs - (optional) plot settings including:
        'Set' - which set to plot ('ti', 'tp', 'y')
        'Unify' - whether to unify sets (default: False)
        Other matplotlib plotting options

Outputs:
    han - handle to the graphics object

Authors: Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Last update: 17-December-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Any


def plot(R, dims: Optional[List[int]] = None, **kwargs) -> Any:
    """
    Plot a projection of the reachable set
    
    Args:
        R: reachSet object
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
    whichset = kwargs.pop('Set', 'ti')  # 'ti', 'tp', or 'y'
    unify = kwargs.pop('Unify', False)
    
    # Check if reachSet is empty
    from .isemptyobject import isemptyobject
    if isemptyobject(R):
        # Plot empty set
        if len(dims) == 2:
            return plt.plot([], [], **kwargs)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            return ax.plot([], [], [], **kwargs)
    
    # Determine which set to plot
    if whichset == 'ti' and 'set' in R.timeInterval:
        sets_to_plot = R.timeInterval['set']
    elif whichset == 'tp' and 'set' in R.timePoint:
        sets_to_plot = R.timePoint['set']
    elif whichset == 'y' and 'set' in R.timeInterval and 'algebraic' in R.timeInterval:
        sets_to_plot = R.timeInterval['algebraic']
    else:
        # Fallback to available sets
        if 'set' in R.timeInterval:
            sets_to_plot = R.timeInterval['set']
        elif 'set' in R.timePoint:
            sets_to_plot = R.timePoint['set']
        else:
            raise ValueError("No sets available for plotting")
    
    if not sets_to_plot:
        # Plot empty set
        if len(dims) == 2:
            return plt.plot([], [], **kwargs)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            return ax.plot([], [], [], **kwargs)
    
    # Plot the sets
    handles = []
    
    for i, s in enumerate(sets_to_plot):
        # Project set to desired dimensions
        if hasattr(s, 'project'):
            s_proj = s.project(dims)
        elif isinstance(s, np.ndarray):
            if s.ndim == 1:
                s_proj = s[dims]
            else:
                s_proj = s[dims, :]
        else:
            continue
        
        # Plot the projected set
        if hasattr(s_proj, 'plot'):
            # Set has its own plot method
            handle = s_proj.plot(**kwargs)
        elif isinstance(s_proj, np.ndarray):
            # Plot as points or lines
            if len(dims) == 2:
                if s_proj.ndim == 1:
                    handle = plt.plot(s_proj[0], s_proj[1], 'o', **kwargs)
                else:
                    handle = plt.plot(s_proj[0, :], s_proj[1, :], **kwargs)
            else:  # 3D
                fig = plt.gcf()
                if not fig.axes or not hasattr(fig.axes[0], 'zaxis'):
                    ax = fig.add_subplot(111, projection='3d')
                else:
                    ax = fig.axes[0]
                
                if s_proj.ndim == 1:
                    handle = ax.plot([s_proj[0]], [s_proj[1]], [s_proj[2]], 'o', **kwargs)
                else:
                    handle = ax.plot(s_proj[0, :], s_proj[1, :], s_proj[2, :], **kwargs)
        else:
            continue
        
        handles.append(handle)
        
        # For subsequent plots, don't show in legend
        if i == 0:
            kwargs['label'] = kwargs.get('label', '_nolegend_')
    
    # Return the first handle or all handles
    if len(handles) == 1:
        return handles[0]
    elif len(handles) > 1:
        return handles
    else:
        return None 