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
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.verbose.plot import read_name_value_pair
from cora_python.g.functions.verbose.plot import read_plot_options
from cora_python.g.functions.verbose.plot import get_unbounded_axis_limits
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

from cora_python.contSet.contSet.contSet import ContSet

def plot(S: ContSet, *args, **kwargs):
    """
    Plot a projection of a contSet
    
    Args:
        S: contSet object
        *args: Variable arguments (dims, linespec, etc.)
        **kwargs: Keyword arguments for plotting options including:
                 purpose: str - plotting purpose ('initialSet', 'reachSet', 'simulation', etc.)
        
    Returns:
        Matplotlib graphics object handle
    """
    # Extract purpose from kwargs if provided
    purpose = kwargs.pop('purpose', 'none')
    
    # Parse input arguments
    S, dims, plot_kwargs, nvpairs_interval, nvpairs_polygon, nvpairs_vertices = _parse_input(S, args, kwargs)
    
    # Process the set and plotting options
    S, plot_kwargs = _process(S, dims, plot_kwargs, purpose)
    
    # Call subfunction depending on number of dimensions to plot
    n = S.dim()
    
    if n == 1:
        # 1D plotting
        handle = S.plot1D(plot_kwargs, nvpairs_interval)
        
    elif n == 2:
        # 2D plotting
        handle = S.plot2D(plot_kwargs, nvpairs_polygon)
        
    elif n == 3:
        # 3D plotting
        handle = S.plot3D(plot_kwargs, nvpairs_vertices)
        
    else:
        # Unable to plot higher-dimensional sets
        raise CORAerror('CORA:plotProperties', f'Cannot plot {n}-dimensional sets')
    
    return handle


def _parse_input(S, args, kwargs):
    """Parse input arguments"""
    
    # Convert args to list
    args_list = list(args)
    
    # Set default values - dims defaults to [0, 1] (Python 0-based indexing)
    defaults = setDefaultValues([[0, 1]], [args_list[0]] if args_list else [])
    dims = defaults[0]
    
    # Convert MATLAB 1-based indexing to Python 0-based if needed
    if isinstance(dims, (list, tuple, np.ndarray)) and len(dims) > 0:
        dims_array = np.array(dims)
        # Detect MATLAB-style indexing: convert when it's clearly MATLAB code
        # MATLAB patterns to convert:
        # - [1] for 1D set -> [0]
        # - [1,2] for 2D set -> [0,1] 
        # - [1,2,3] for 3D set -> [0,1,2]
        # - [1,3] for 3D+ set -> [0,2] (partial projection)
        # Don't convert if dimensions are out of valid MATLAB range
        is_single_dim_matlab = (len(dims_array) == 1 and dims_array[0] == 1)  # [1] -> [0]
        is_consecutive_from_1 = (len(dims_array) > 1 and 
                                np.all(dims_array == np.arange(1, len(dims_array) + 1)))
        is_full_matlab_dims = (len(dims_array) == S.dim() and is_consecutive_from_1)
        is_partial_matlab_dims = (len(dims_array) > 1 and 
                                 np.all(dims_array >= 1) and 
                                 np.max(dims_array) <= S.dim())
        
        if is_single_dim_matlab or is_full_matlab_dims or is_partial_matlab_dims:
            # Convert to 0-based indexing
            dims = dims_array - 1
        dims = dims.tolist() if hasattr(dims, 'tolist') else dims
    
    # Check input arguments
    inputArgsCheck([
        [S, 'att', 'contSet'],
        [dims, 'att', 'numeric', ['nonempty', 'integer', 'nonnegative', 'vector']]
    ])
    
    # Additional validation: check if dimensions are valid for the set
    set_dim = S.dim()
    for d in dims:
        if d < 0 or d >= set_dim:
            raise CORAerror('CORA:wrongInput', f'Dimension {d+1} does not exist (set has {set_dim} dimensions)')
    
    # Check dimension constraints
    if len(dims) < 1:
        raise CORAerror('CORA:plotProperties', 'At least one dimension must be specified')
    elif len(dims) > 3:
        raise CORAerror('CORA:plotProperties', 'Cannot plot more than 3 dimensions')
    
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


def _process(S, dims, plot_kwargs, purpose='none'):
    """Process set and plotting options"""
    
    # Project set to requested dimensions
    S = S.project(dims)
    
    # Check if set is bounded
    if hasattr(S, '__class__') and S.__class__.__name__ == 'Interval':
        I = S
    else:
        I = S.interval()

    if not I.is_bounded():
        # Intersect with current plot axis for unbounded sets
        S = _intersect_with_axis_limits(S, plot_kwargs)
        
        # Fill unbounded sets (fullspace, halfspace, etc.)
        if purpose == 'none':  # Don't override reachSet purpose
            purpose = 'fill'
    
    # Apply purpose-specific plot options
    plot_kwargs = read_plot_options(plot_kwargs, purpose)
    
    return S, plot_kwargs


def _intersect_with_axis_limits(S, plot_kwargs):
    """Intersect unbounded sets with axis limits"""
    from cora_python.contSet.interval.interval import Interval
    
    # Get quick estimate of vertices from interval hull
    if isinstance(S, Interval):
        I = S
    else:
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
        I_axis = Interval(np.array([xlim[0], ylim[0]]), 
                         np.array([xlim[1], ylim[1]]))
        
    elif n == 3:
        # 3D case
        xlim, ylim, zlim = get_unbounded_axis_limits(V)
        
        # Create axis interval
        I_axis = Interval(np.array([xlim[0], ylim[0], zlim[0]]), 
                         np.array([xlim[1], ylim[1], zlim[1]]))
    else:
        raise CORAerror('CORA:plotProperties', f'Cannot handle {n}-dimensional plotting')
    
    # Project to given dimensions (use 0-based indexing)
    I_axis = I_axis.project(list(range(S.dim())))
    
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