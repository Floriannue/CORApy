"""
plotOverTime - plots a projection of the specification over time

Syntax:
    han = plotOverTime(spec)
    han = plotOverTime(spec, dims)
    han = plotOverTime(spec, dims, **plot_options)

Inputs:
    spec - specification object
    dims - (optional) dimension for projection (single integer)
    **plot_options - (optional) plot settings (matplotlib options)

Outputs:
    han - handle to the graphics object

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 03-March-2023 (MATLAB)
Last update: 12-July-2023 (TL, restructure) (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Any, Union
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def plotOverTime(specs, dims: Optional[int] = None, **plot_options):
    """
    Plot a projection of the specification over time
    
    Args:
        specs: specification object or list of specification objects
        dims: dimension for projection (default: 0, single integer for time plot)
        **plot_options: plot settings for matplotlib
        
    Returns:
        han: handle to the graphics object(s)
    """
    
    # 1. parse input
    specs, dims, plot_opts = _parse_input(specs, dims, plot_options)

    # 2. preprocess
    spectime, hold_status = _preprocess(specs)

    # 3. plot specifications
    han = _plot_specs(specs, dims, plot_opts, spectime)

    # 4. postprocess
    _postprocess(hold_status)

    return han


def _parse_input(specs, dims, plot_options):
    """Parse and validate input arguments"""
    
    # Convert single spec to list
    if not isinstance(specs, list):
        specs = [specs]
    
    # Check that all items are specification objects
    for spec in specs:
        if not hasattr(spec, 'type') or not hasattr(spec, 'set'):
            raise CORAError('CORA:wrongInput',
                'Input must be specification object(s)')
    
    # Set default dimension
    if dims is None:
        dims = 0  # Python uses 0-based indexing, plot first dimension vs time
    
    # Validate dimension
    if not isinstance(dims, int):
        raise CORAError('CORA:wrongInput',
            'Dimension must be a single integer for time plots')
    
    if dims < 0:
        raise CORAError('CORA:wrongInput',
            'Dimension must be non-negative integer')
    
    return specs, dims, plot_options


def _preprocess(specs):
    """Preprocess plotting - handle time frames and hold status"""
    
    # Check if any of the specifications have a defined time frame
    spectime = []
    no_time_given = []
    
    for spec in specs:
        if spec.time is None or _represents_empty_set(spec.time):
            spectime.append(None)
            no_time_given.append(True)
        else:
            spectime.append(spec.time)
            no_time_given.append(False)
    
    # Check if any specifications have time defined
    if all(no_time_given):
        raise CORAError('CORA:specialError',
            'No specification has a defined time frame.')
    
    # Set empty .time fields to min/max of others
    if any(no_time_given):
        t_min = float('inf')
        t_max = float('-inf')
        
        for i, spec in enumerate(specs):
            if not no_time_given[i] and spectime[i] is not None:
                if hasattr(spectime[i], 'infimum') and hasattr(spectime[i], 'supremum'):
                    t_min = min(t_min, spectime[i].infimum())
                    t_max = max(t_max, spectime[i].supremum())
                elif hasattr(spectime[i], 'inf') and hasattr(spectime[i], 'sup'):
                    t_min = min(t_min, spectime[i].inf)
                    t_max = max(t_max, spectime[i].sup)
                elif isinstance(spectime[i], (list, tuple)) and len(spectime[i]) == 2:
                    t_min = min(t_min, spectime[i][0])
                    t_max = max(t_max, spectime[i][1])
        
        # Create default time interval for specs without time
        default_time = _create_interval(t_min, t_max)
        for i in range(len(spectime)):
            if no_time_given[i]:
                spectime[i] = default_time
    
    # Check if we're adding to existing plot
    ax = plt.gca()
    hold_status = len(ax.get_children()) > 0
    
    if not hold_status:
        # Clear plot if not holding
        plt.clf()
    
    return spectime, hold_status


def _plot_specs(specs, dims, plot_opts, spectime):
    """Plot specifications over time, grouped by type"""
    
    # Group specifications by type
    spec_groups = {}
    time_groups = {}
    
    for i, spec in enumerate(specs):
        spec_type = spec.type
        if spec_type not in spec_groups:
            spec_groups[spec_type] = []
            time_groups[spec_type] = []
        spec_groups[spec_type].append(spec)
        time_groups[spec_type].append(spectime[i])
    
    handles = []
    
    for spec_type, spec_list in spec_groups.items():
        
        # Get type-specific plot options
        if spec_type in ['safeSet', 'unsafeSet', 'invariant']:
            type_opts = _get_spec_plot_options(spec_type, plot_opts)
        elif spec_type == 'custom':
            type_opts = plot_opts.copy()
        else:
            raise CORAError('CORA:notSupported',
                f"Plotting specifications of type '{spec_type}' over time is not yet supported.")
        
        # Create time-extended sets
        sets = []
        for i, spec in enumerate(spec_list):
            time_interval = time_groups[spec_type][i]
            
            # Project the specification set to the desired dimension
            projected_set = _project_set(spec.set, dims)
            
            # Create cartesian product of time interval and projected set
            time_set = _create_cartesian_product(time_interval, projected_set)
            sets.append(time_set)
        
        # Plot all sets of this type
        han = _plot_multiple_sets_as_one(sets, [0, 1], type_opts)  # time vs dimension
        handles.extend(han if isinstance(han, list) else [han])
    
    return handles


def _get_spec_plot_options(spec_type, base_opts):
    """Get plotting options specific to specification type"""
    
    # Default colors and styles for different specification types
    defaults = {
        'safeSet': {'color': 'green', 'alpha': 0.3, 'label': 'Safe Set'},
        'unsafeSet': {'color': 'red', 'alpha': 0.3, 'label': 'Unsafe Set'},
        'invariant': {'color': 'blue', 'alpha': 0.3, 'label': 'Invariant Set'}
    }
    
    # Start with type-specific defaults
    opts = defaults.get(spec_type, {}).copy()
    
    # Override with user-provided options
    opts.update(base_opts)
    
    return opts


def _project_set(set_obj, dim):
    """Project a set to the specified dimension"""
    
    if hasattr(set_obj, 'project'):
        return set_obj.project([dim])
    elif hasattr(set_obj, 'interval') and hasattr(set_obj, 'generators'):
        # For zonotope-like objects, extract the dimension
        if hasattr(set_obj, 'center'):
            center = set_obj.center()
            if len(center) > dim:
                # Simple interval approximation
                return _create_interval(center[dim] - 0.1, center[dim] + 0.1)
    
    # Fallback: return a default interval
    return _create_interval(-1, 1)


def _create_interval(inf_val, sup_val):
    """Create an interval object"""
    
    try:
        # Try to use the interval class if available
        from cora_python.contSet.interval import Interval
        return Interval([inf_val], [sup_val])
    except ImportError:
        # Fallback: return as tuple
        return (inf_val, sup_val)


def _create_cartesian_product(time_interval, projected_set):
    """Create cartesian product of time interval and projected set"""
    
    # Extract time bounds
    if hasattr(time_interval, 'infimum') and hasattr(time_interval, 'supremum'):
        t_min = time_interval.infimum()
        t_max = time_interval.supremum()
    elif hasattr(time_interval, 'inf') and hasattr(time_interval, 'sup'):
        t_min = time_interval.inf
        t_max = time_interval.sup
    elif isinstance(time_interval, (list, tuple)) and len(time_interval) == 2:
        t_min, t_max = time_interval
    else:
        t_min, t_max = 0, 1  # Default
    
    # Extract set bounds
    if hasattr(projected_set, 'infimum') and hasattr(projected_set, 'supremum'):
        x_min = projected_set.infimum()
        x_max = projected_set.supremum()
    elif hasattr(projected_set, 'inf') and hasattr(projected_set, 'sup'):
        x_min = projected_set.inf
        x_max = projected_set.sup
    elif isinstance(projected_set, (list, tuple)) and len(projected_set) == 2:
        x_min, x_max = projected_set
    else:
        x_min, x_max = -1, 1  # Default
    
    # Return as a rectangle in time-space coordinates
    return {
        'time_bounds': (t_min, t_max),
        'space_bounds': (x_min, x_max)
    }


def _plot_multiple_sets_as_one(sets, dims, plot_opts):
    """Plot multiple time-extended sets"""
    
    handles = []
    
    for i, set_data in enumerate(sets):
        try:
            # Adjust options for multiple sets
            set_opts = plot_opts.copy()
            if i > 0 and 'label' in set_opts:
                # Only label first set to avoid legend clutter
                del set_opts['label']
            
            # Extract bounds
            t_min, t_max = set_data['time_bounds']
            x_min, x_max = set_data['space_bounds']
            
            # Create rectangle
            from matplotlib.patches import Rectangle
            rect = Rectangle((t_min, x_min), t_max - t_min, x_max - x_min,
                           facecolor=set_opts.get('color', 'blue'),
                           alpha=set_opts.get('alpha', 0.3),
                           label=set_opts.get('label', None))
            
            ax = plt.gca()
            ax.add_patch(rect)
            handles.append(rect)
            
        except Exception as e:
            print(f"Warning: Could not plot set {i}: {e}")
    
    return handles


def _postprocess(hold_status):
    """Postprocess plotting"""
    
    # Add grid and labels for better visualization
    plt.grid(True, alpha=0.3)
    
    # Set axis labels
    ax = plt.gca()
    if ax.get_xlabel() == '':
        plt.xlabel('Time')
    if ax.get_ylabel() == '':
        plt.ylabel('x₀')
    
    # Add legend if there are labeled items
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        plt.legend()


def _represents_empty_set(obj):
    """Check if object represents an empty set"""
    if obj is None:
        return True
    if hasattr(obj, 'representsa_'):
        return obj.representsa_('emptySet', 1e-10)
    return False 