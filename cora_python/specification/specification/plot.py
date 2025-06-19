"""
plot - plots a projection of the specification

Syntax:
    han = plot(spec)
    han = plot(spec, dims)
    han = plot(spec, dims, **plot_options)

Inputs:
    spec - specification object
    dims - (optional) dimensions for projection
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


def plot(specs, dims: Optional[List[int]] = None, **plot_options):
    """
    Plot a projection of the specification
    
    Args:
        specs: specification object or list of specification objects
        dims: dimensions for projection (default: [0, 1])
        **plot_options: plot settings for matplotlib
        
    Returns:
        han: handle to the graphics object(s)
    """
    
    # 1. parse input
    specs, dims, plot_opts = _parse_input(specs, dims, plot_options)

    # 2. preprocess
    hold_status = _preprocess()

    # 3. plot specifications
    han = _plot_specs(specs, dims, plot_opts)

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
    
    # Set default dimensions
    if dims is None:
        dims = [0, 1]  # Python uses 0-based indexing
    
    # Validate dimensions
    if not isinstance(dims, (list, np.ndarray)):
        dims = [dims]
    
    dims = np.array(dims, dtype=int)
    
    if len(dims) < 2:
        raise CORAError('CORA:plotProperties', 
            'At least 2 dimensions required for plotting')
    elif len(dims) > 3:
        raise CORAError('CORA:plotProperties',
            'At most 3 dimensions supported for plotting')
    
    # Validate that dimensions are non-negative integers
    if np.any(dims < 0):
        raise CORAError('CORA:wrongInput',
            'Dimensions must be non-negative integers')
    
    return specs, dims, plot_options


def _preprocess():
    """Preprocess plotting - handle hold status"""
    
    # Check if we're adding to existing plot
    ax = plt.gca()
    hold_status = len(ax.get_children()) > 0
    
    if not hold_status:
        # Clear plot if not holding
        plt.clf()
    
    return hold_status


def _plot_specs(specs, dims, plot_opts):
    """Plot specifications, grouped by type"""
    
    # Group specifications by type
    spec_groups = {}
    for spec in specs:
        spec_type = spec.type
        if spec_type not in spec_groups:
            spec_groups[spec_type] = []
        spec_groups[spec_type].append(spec)
    
    handles = []
    
    for spec_type, spec_list in spec_groups.items():
        
        # Get type-specific plot options
        if spec_type in ['safeSet', 'unsafeSet', 'invariant']:
            type_opts = _get_spec_plot_options(spec_type, plot_opts)
        elif spec_type == 'custom':
            type_opts = plot_opts.copy()
        else:
            raise CORAError('CORA:notSupported',
                f"Plotting specifications of type '{spec_type}' is not yet supported.")
        
        # Plot all sets of this type
        sets_to_plot = [spec.set for spec in spec_list]
        han = _plot_multiple_sets_as_one(sets_to_plot, dims, type_opts)
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


def _plot_multiple_sets_as_one(sets, dims, plot_opts):
    """Plot multiple sets as unified visualization"""
    
    handles = []
    
    for i, set_obj in enumerate(sets):
        if hasattr(set_obj, 'plot'):
            # Use the set's own plot method if available
            try:
                # Adjust options for multiple sets
                set_opts = plot_opts.copy()
                if i > 0 and 'label' in set_opts:
                    # Only label first set to avoid legend clutter
                    del set_opts['label']
                
                han = set_obj.plot(dims, **set_opts)
                handles.append(han)
                
            except Exception as e:
                print(f"Warning: Could not plot set {i}: {e}")
        else:
            print(f"Warning: Set {i} does not have a plot method")
    
    return handles


def _postprocess(hold_status):
    """Postprocess plotting"""
    
    # Add grid and labels for better visualization
    plt.grid(True, alpha=0.3)
    
    # Set axis labels
    ax = plt.gca()
    if ax.get_xlabel() == '':
        plt.xlabel('x₀')
    if ax.get_ylabel() == '':
        plt.ylabel('x₁')
    
    # Add legend if there are labeled items
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        plt.legend()
    
    # Ensure equal aspect ratio for better visualization
    plt.axis('equal') 