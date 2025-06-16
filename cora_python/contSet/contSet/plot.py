"""
plot - plots a projection of a contSet

This function visualizes a 2-dimensional projection of the boundary of a set.
It delegates to appropriate dimension-specific plotting functions.

Syntax:
    handle = plot(S)
    handle = plot(S, dims)
    handle = plot(S, dims, linespec)
    handle = plot(S, dims, **kwargs)

Inputs:
    S - contSet object
    dims - (optional) dimensions for projection (default: [1, 2])
    linespec - (optional) line specifications (e.g., '--*r')
    kwargs - (optional) plot settings as keyword arguments

Outputs:
    handle - matplotlib graphics object handle

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 14-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import List, Any, Optional, Union
from cora_python.g import (set_default_values, input_args_check, read_name_value_pair, 
                          read_plot_options, get_unbounded_axis_limits, CORAError)


def plot(S, *args, **kwargs):
    """
    Plot a projection of a contSet
    
    Args:
        S: contSet object
        *args: Variable arguments (dims, linespec, etc.)
        **kwargs: Keyword arguments for plotting options
        
    Returns:
        Matplotlib graphics object handle
    """
    # Parse input arguments
    S, dims, plot_kwargs, nvpairs_interval, nvpairs_polygon, nvpairs_vertices = _parse_input(S, args, kwargs)
    
    # Process the set and plotting options
    S, plot_kwargs = _process(S, dims, plot_kwargs)
    
    # Call subfunction depending on number of dimensions to plot
    n = S.dim()
    
    if n == 1:
        # 1D plotting
        from .plot1D import plot1D
        handle = plot1D(S, plot_kwargs, nvpairs_interval)
        
    elif n == 2:
        # 2D plotting
        from .plot2D import plot2D
        handle = plot2D(S, plot_kwargs, nvpairs_polygon)
        
    elif n == 3:
        # 3D plotting
        from .plot3D import plot3D
        handle = plot3D(S, plot_kwargs, nvpairs_vertices)
        
    else:
        # Unable to plot higher-dimensional sets
        raise CORAError('CORA:plotProperties', f'Cannot plot {n}-dimensional sets')
    
    return handle


def _parse_input(S, args, kwargs):
    """Parse input arguments"""
    
    # Convert args to list
    args_list = list(args)
    
    # Set default values - dims defaults to [0, 1] (Python 0-based indexing)
    dims = set_default_values([[0, 1]], [args_list[0]] if args_list else [])
    dims = dims[0]
    
    # Convert MATLAB 1-based indexing to Python 0-based if needed
    if isinstance(dims, (list, tuple, np.ndarray)) and len(dims) > 0:
        # Check if dims looks like MATLAB 1-based indexing (all values >= 1)
        dims_array = np.array(dims)
        if np.all(dims_array >= 1):
            # Convert to 0-based indexing
            dims = dims_array - 1
        dims = dims.tolist() if hasattr(dims, 'tolist') else dims
    
    # Check input arguments
    input_args_check([
        [S, 'att', 'contSet'],
        [dims, 'att', 'numeric', {'nonempty': True, 'integer': True, 'positive': True, 'vector': True}]
    ])
    
    # Check dimension constraints
    if len(dims) < 1:
        raise CORAError('CORA:plotProperties', 'At least one dimension must be specified')
    elif len(dims) > 3:
        raise CORAError('CORA:plotProperties', 'Cannot plot more than 3 dimensions')
    
    # Remove dims from args if it was provided
    if args_list:
        args_list = args_list[1:]
    
    # Parse plot options from remaining args and kwargs
    all_options = args_list + [item for pair in kwargs.items() for item in pair]
    plot_kwargs = read_plot_options(all_options)
    
    # Read additional parameters based on set type
    nvpairs_interval = []
    nvpairs_polygon = []
    nvpairs_vertices = []
    
    if hasattr(S, '__class__') and S.__class__.__name__ == 'polyZonotope':
        remaining_opts, splits = read_name_value_pair(all_options, 'Splits', 'isscalar', 8)
        nvpairs_interval = ['split', splits]
        nvpairs_polygon = [splits]
        nvpairs_vertices = [splits]
        
    elif hasattr(S, '__class__') and S.__class__.__name__ == 'conPolyZono':
        remaining_opts, splits = read_name_value_pair(all_options, 'Splits', 'isscalar', 8)
        nvpairs_interval = ['conZonotope']
        nvpairs_polygon = [splits]
        nvpairs_vertices = [splits]
    
    return S, dims, plot_kwargs, nvpairs_interval, nvpairs_polygon, nvpairs_vertices


def _process(S, dims, plot_kwargs):
    """Process set and plotting options"""
    
    # Project set to requested dimensions
    S = S.project(dims)
    
    # Determine purpose for plot options
    purpose = 'none'
    
    # Check if set is bounded
    I = S.interval()
    if not I.is_bounded():
        # Intersect with current plot axis for unbounded sets
        S = _intersect_with_axis_limits(S, plot_kwargs)
        
        # Fill unbounded sets (fullspace, halfspace, etc.)
        purpose = 'fill'
    
    # Apply purpose-specific plot options
    plot_kwargs = read_plot_options([], purpose)
    
    return S, plot_kwargs


def _intersect_with_axis_limits(S, plot_kwargs):
    """Intersect unbounded sets with axis limits"""
    
    # Get quick estimate of vertices from interval hull
    I = S.interval()
    V = I.vertices_()
    
    # Consider given positioning options
    V = _position_vertices(V, plot_kwargs)
    n, m = V.shape
    
    # Get axis limits
    if n <= 2:
        # 1D-2D case
        xlim, ylim = get_unbounded_axis_limits(V)
        
        # Create axis interval
        from cora_python.contSet.interval.interval import interval
        I_axis = interval(np.array([xlim[0], ylim[0]]), 
                         np.array([xlim[1], ylim[1]]))
        
    elif n == 3:
        # 3D case
        xlim, ylim, zlim = get_unbounded_axis_limits(V)
        
        # Create axis interval
        from cora_python.contSet.interval.interval import interval
        I_axis = interval(np.array([xlim[0], ylim[0], zlim[0]]), 
                         np.array([xlim[1], ylim[1], zlim[1]]))
    else:
        raise CORAError('CORA:plotProperties', f'Cannot handle {n}-dimensional plotting')
    
    # Project to given dimensions
    I_axis = I_axis.project(list(range(1, S.dim() + 1)))
    
    # Intersect with set
    S = S.and_(I_axis, 'exact')
    
    return S


def _position_vertices(V, plot_kwargs):
    """Position vertices according to XPos, YPos, ZPos options"""
    
    n, N = V.shape
    z = np.zeros(N)
    
    # Read positioning options
    remaining_opts, x_pos = read_name_value_pair(list(plot_kwargs.items()), 'XPos', ['isnumeric', 'isscalar'])
    remaining_opts, y_pos = read_name_value_pair(remaining_opts, 'YPos', ['isnumeric', 'isscalar'])
    remaining_opts, z_pos = read_name_value_pair(remaining_opts, 'ZPos', ['isnumeric', 'isscalar'])
    
    # Build new vertex matrix
    new_V = []
    
    if x_pos is not None:
        new_V.append(z.copy())
        n += 1
    
    if n >= 1:
        new_V.append(V[0, :])
    
    if y_pos is not None:
        if len(new_V) == 1:  # Only x_pos was added
            new_V.append(z.copy())
        else:
            new_V.insert(-1, z.copy())
        n += 1
        
    if n >= 2:
        new_V.append(V[1, :])
    
    if z_pos is not None:
        if n <= 2:
            new_V.append(z.copy())
        else:
            new_V.append(V[2, :])
    
    if n >= 3:
        new_V.append(V[2, :])
    
    return np.array(new_V) if new_V else V 