"""
plot - plots a projection of the simulated trajectories

Syntax:
    han = plot(simRes)
    han = plot(simRes, dims)
    han = plot(simRes, dims, **kwargs)

Inputs:
    simRes - simResult object or list of simResult objects
    dims - (optional) dimensions for projection (default: [0, 1])
    **kwargs - (optional) plot settings including:
        'Traj' - which trajectory to plot ('x', 'y', 'a')
        'DisplayName' - label for legend
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
from typing import List, Optional, Any, Union


def plot(simRes: Union['SimResult', List['SimResult']], dims: Optional[List[int]] = None, **kwargs) -> Any:
    """
    Plot a projection of the simulated trajectories
    
    Args:
        simRes: simResult object or list of simResult objects
        dims: Dimensions for projection (default: [0, 1])
        **kwargs: Plot settings including:
            'DisplayName': label for legend (MATLAB style)
            'Traj': which trajectory to plot ('x', 'y', 'a')
            Other matplotlib options
        
    Returns:
        Handle to the graphics object
    """
    # Handle DisplayName parameter (MATLAB style)
    display_name = kwargs.pop('DisplayName', None)
    if display_name is not None:
        kwargs['label'] = display_name
    
    # Set default dimensions
    if dims is None:
        dims = [0, 1]
    
    # Validate inputs
    if len(dims) < 2:
        raise ValueError("At least 2 dimensions required for plotting")
    elif len(dims) > 3:
        raise ValueError("At most 3 dimensions supported for plotting")
    
    # Handle both single SimResult and list of SimResults
    if isinstance(simRes, list):
        return _plot_simres_list(simRes, dims, **kwargs)
    else:
        return _plot_single_simres(simRes, dims, **kwargs)


def _plot_simres_list(simRes_list: List['SimResult'], dims: List[int], **kwargs) -> Any:
    """Plot a list of SimResult objects"""
    
    # Check if any simResult is non-empty
    has_data = False
    for sr in simRes_list:
        from .isemptyobject import isemptyobject
        if not isemptyobject(sr):
            has_data = True
            break
    
    if not has_data:
        # Plot empty trajectory
        if len(dims) == 2:
            return plt.plot([], [], **kwargs)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            return ax.plot([], [], [], **kwargs)
    
    # Parse options
    whichtraj = kwargs.pop('Traj', 'x')  # 'x', 'y', or 'a'
    
    # Collect all trajectories from all simResult objects
    all_handles = []
    first_plot = True
    
    for sr_idx, sr in enumerate(simRes_list):
        from .isemptyobject import isemptyobject
        if isemptyobject(sr):
            continue
        
        # Get trajectories from this simResult
        trajs = _get_trajectories(sr, whichtraj)
        if not trajs:
            continue
        
        # Plot each trajectory
        for traj_idx, traj in enumerate(trajs):
            if traj.size == 0:
                continue
            
            # Extract desired dimensions
            traj_proj = _extract_dimensions(traj, dims)
            if traj_proj is None:
                continue
            
            # Set up kwargs for this trajectory
            current_kwargs = kwargs.copy()
            
            # Only set label for the first trajectory to avoid legend clutter
            if first_plot and 'label' in kwargs:
                current_kwargs['label'] = kwargs['label']
                first_plot = False
            elif 'label' in current_kwargs:
                # Remove label for subsequent trajectories to avoid clutter
                del current_kwargs['label']
            
            # Plot the trajectory
            handle = _plot_trajectory(traj_proj, dims, **current_kwargs)
            if handle:
                all_handles.append(handle)
    
    # Return handles
    if len(all_handles) == 1:
        return all_handles[0]
    elif len(all_handles) > 1:
        return all_handles
    else:
        return None


def _plot_single_simres(simRes: 'SimResult', dims: List[int], **kwargs) -> Any:
    """Plot a single SimResult object"""
    
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
    
    # Get trajectories to plot
    trajs_to_plot = _get_trajectories(simRes, whichtraj)
    
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
    first_plot = True
    
    for i, traj in enumerate(trajs_to_plot):
        if traj.size == 0:
            continue
        
        # Extract the desired dimensions
        traj_proj = _extract_dimensions(traj, dims)
        if traj_proj is None:
            continue
        
        # Set up kwargs for this trajectory
        current_kwargs = kwargs.copy()
        
        # Only set label for the first trajectory
        if not first_plot and 'label' in current_kwargs:
            del current_kwargs['label']
        first_plot = False
        
        # Plot the trajectory
        handle = _plot_trajectory(traj_proj, dims, **current_kwargs)
        if handle:
            handles.append(handle)
    
    # Return the first handle or all handles
    if len(handles) == 1:
        return handles[0]
    elif len(handles) > 1:
        return handles
    else:
        return None


def _get_trajectories(simRes: 'SimResult', whichtraj: str) -> List[np.ndarray]:
    """Get the appropriate trajectories from a SimResult object"""
    if whichtraj == 'x':
        return simRes.x
    elif whichtraj == 'y':
        if not simRes.y:
            raise ValueError("No output trajectories available")
        return simRes.y
    elif whichtraj == 'a':
        if not simRes.a:
            raise ValueError("No algebraic trajectories available")
        return simRes.a
    else:
        raise ValueError(f"Unknown trajectory type: {whichtraj}")


def _extract_dimensions(traj: np.ndarray, dims: List[int]) -> Optional[np.ndarray]:
    """Extract desired dimensions from trajectory"""
    if traj.ndim == 1:
        # Single point trajectory
        if len(dims) <= len(traj) and max(dims) < len(traj):
            return traj[dims]
        else:
            return None
    else:
        # Multi-point trajectory
        if traj.shape[1] > max(dims):
            return traj[:, dims]
        else:
            return None


def _plot_trajectory(traj_proj: np.ndarray, dims: List[int], **kwargs) -> Any:
    """Plot a single trajectory projection"""
    
    # Apply simulation purpose for color defaults
    from cora_python.g.functions.verbose.plot.read_plot_options import read_plot_options
    kwargs = read_plot_options(kwargs, purpose='simulation')
    
    # Set default alpha
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.8  # Higher alpha for better visibility
    
    # Set default line style
    if 'linestyle' not in kwargs and 'ls' not in kwargs:
        kwargs['linestyle'] = '-'
    
    # Set default line width - slightly thicker for smoother appearance
    if 'linewidth' not in kwargs and 'lw' not in kwargs:
        kwargs['linewidth'] = 1.0
    
    # Set z-order for simulations (highest - on top like MATLAB)
    if 'zorder' not in kwargs:
        kwargs['zorder'] = 10
    
    # Enhanced smoothing settings for MATLAB-like appearance
    if 'antialiased' not in kwargs:
        kwargs['antialiased'] = True
    if 'solid_capstyle' not in kwargs:
        kwargs['solid_capstyle'] = 'round'
    if 'solid_joinstyle' not in kwargs:
        kwargs['solid_joinstyle'] = 'round'
    
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
    
    return handle 