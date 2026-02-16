"""
read_plot_options - reads and processes plot options

This function mimics MATLAB's readPlotOptions functionality for processing
LineSpecification and name-value pairs for matplotlib plotting.

Syntax:
    plot_kwargs = read_plot_options(plot_options, purpose='none')

Inputs:
    plot_options - list of LineSpec + Name-Value pairs
    purpose - plot purpose ('fill', 'none', etc.)

Outputs:
    plot_kwargs - dictionary of matplotlib plotting options

Authors: Mark Wetzlinger, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.colors as mcolors
from typing import List, Any, Dict, Optional
from .color.default_plot_color import default_plot_color
from .color.next_color import next_color
from .color.cora_color import cora_color


def read_plot_options(plot_options, purpose: str = 'none') -> Dict[str, Any]:
    """
    Read and process plot options for matplotlib
    
    Args:
        plot_options: List of LineSpec + Name-Value pairs, or Dict of options
        purpose: Plot purpose ('fill', 'none', etc.)
        
    Returns:
        Dictionary of matplotlib plotting options
    """
    # Handle both list and dict inputs
    if isinstance(plot_options, dict):
        plot_kwargs = plot_options.copy()
        # Convert MATLAB parameter names in dict input
        matlab_to_mpl = {
            'Color': 'color',
            'LineWidth': 'linewidth', 
            'LineStyle': 'linestyle',
            'Marker': 'marker',
            'MarkerSize': 'markersize',
            'FaceColor': 'facecolor',
            'EdgeColor': 'edgecolor',
            'FaceAlpha': 'alpha',
            'EdgeAlpha': 'alpha',
            'DisplayName': 'label',  # Key fix for dict inputs
            'HandleVisibility': None,
            'Unify': 'Unify',  # Keep Unify as-is for reachSet plotting
            'Set': 'Set',  # Keep Set as-is for reachSet plotting
            'XPos': 'XPos',  # Keep positioning parameters as-is
            'YPos': 'YPos',  # Keep positioning parameters as-is
            'ZPos': 'ZPos'   # Keep positioning parameters as-is
        }
        for matlab_name, mpl_name in matlab_to_mpl.items():
            if matlab_name in plot_kwargs and mpl_name is not None:
                plot_kwargs[mpl_name] = plot_kwargs.pop(matlab_name)
        
        plot_options = []  # Empty list for linespec parsing
    else:
        if not plot_options:
            plot_options = []
        
        # Initialize with default options
        plot_kwargs = {}
        
        # Parse linespec if provided
        linespec_kwargs = {}
        if plot_options and isinstance(plot_options, list) and len(plot_options) > 0 and isinstance(plot_options[0], str):
            # Check if first element is actually a linespec and not a parameter name
            first_elem = plot_options[0]
            # Known MATLAB parameter names that are NOT linespec
            matlab_param_names = {'Color', 'LineWidth', 'LineStyle', 'Marker', 'MarkerSize', 
                                'FaceColor', 'EdgeColor', 'FaceAlpha', 'EdgeAlpha', 
                                'DisplayName', 'HandleVisibility', 'Unify', 'Set', 
                                'XPos', 'YPos', 'ZPos'}
            
            if first_elem not in matlab_param_names:
                linespec_kwargs = _parse_linespec(first_elem)
                plot_options = plot_options[1:]  # Remove linespec from options
        
        # Parse name-value pairs
        nv_kwargs = _parse_name_value_pairs(plot_options)
        
        # Merge linespec and name-value pairs (name-value pairs take precedence)
        plot_kwargs.update(linespec_kwargs)
        plot_kwargs.update(nv_kwargs)
    
    # Apply purpose-specific defaults
    plot_kwargs = _apply_purpose_defaults(plot_kwargs, purpose)
    
    # Ensure we have color if not specified
    has_face_color = ('facecolor' in plot_kwargs or 'FaceColor' in plot_kwargs)
    if 'color' not in plot_kwargs and not has_face_color:
        c = next_color()
        plot_kwargs['color'] = c
        # Also set facecolor so 2D filled plots (intervals, zonotopes, etc.) get automatic cycle color
        plot_kwargs['facecolor'] = c
    # Remove conflicting color if facecolor is specified for filled plots
    elif has_face_color and 'color' in plot_kwargs:
        # For filled plots, facecolor takes precedence over color
        del plot_kwargs['color']
    
    return plot_kwargs


def _parse_linespec(linespec: str) -> Dict[str, Any]:
    """Parse MATLAB-style linespec string"""
    kwargs = {}
    
    # Color mapping
    color_map = {
        'b': 'blue', 'g': 'green', 'r': 'red', 'c': 'cyan',
        'm': 'magenta', 'y': 'yellow', 'k': 'black', 'w': 'white'
    }
    
    # Marker mapping
    marker_map = {
        'o': 'o', '+': '+', '*': '*', '.': '.', 'x': 'x',
        '_': '_', '|': '|', 's': 's', 'd': 'D', '^': '^',
        'v': 'v', '<': '<', '>': '>', 'p': 'p', 'h': 'h'
    }
    
    # Line style mapping
    linestyle_map = {
        '-': '-', '--': '--', '-.': '-.', ':': ':'
    }
    
    # Parse color
    for char, color in color_map.items():
        if char in linespec:
            kwargs['color'] = color
            linespec = linespec.replace(char, '')
            break
    
    # Parse marker
    for char, marker in marker_map.items():
        if char in linespec:
            kwargs['marker'] = marker
            linespec = linespec.replace(char, '')
            break
    
    # Parse line style
    for style_str, style in linestyle_map.items():
        if style_str in linespec:
            kwargs['linestyle'] = style
            linespec = linespec.replace(style_str, '')
            break
    
    return kwargs


def _parse_name_value_pairs(options: List[Any]) -> Dict[str, Any]:
    """Parse name-value pairs"""
    kwargs = {}
    
    # Convert MATLAB name-value pairs to matplotlib kwargs
    matlab_to_mpl = {
        'Color': 'color',
        'LineWidth': 'linewidth',
        'LineStyle': 'linestyle',
        'Marker': 'marker',
        'MarkerSize': 'markersize',
        'FaceColor': 'facecolor',
        'EdgeColor': 'edgecolor',
        'FaceAlpha': 'alpha',
        'EdgeAlpha': 'alpha',  # Note: matplotlib has single alpha
        'DisplayName': 'label',  # This is the key MATLAB->Python mapping
        'HandleVisibility': None,  # Not directly supported in matplotlib
        'Unify': 'Unify',  # Keep Unify as-is for reachSet plotting
        'Set': 'Set',  # Keep Set as-is for reachSet plotting
        'XPos': 'XPos',  # Keep positioning parameters as-is
        'YPos': 'YPos',  # Keep positioning parameters as-is
        'ZPos': 'ZPos'   # Keep positioning parameters as-is
    }
    
    i = 0
    while i < len(options) - 1:
        if isinstance(options[i], str):
            name = options[i]
            value = options[i + 1]
            
            # Convert MATLAB parameter name to matplotlib
            mpl_name = matlab_to_mpl.get(name, name.lower())
            
            if mpl_name is not None:
                # Process value if needed
                if name == 'Color' or name == 'FaceColor' or name == 'EdgeColor':
                    value = _process_color_value(value)
                
                kwargs[mpl_name] = value
            
            i += 2
        else:
            i += 1
    
    return kwargs


def _process_color_value(color_value: Any) -> Any:
    """Process color value for matplotlib"""
    if isinstance(color_value, str):
        if color_value == 'next':
            return next_color()
        elif color_value == 'default':
            return default_plot_color()
        elif color_value.startswith('CORA:'):
            # Handle CORA color identifiers
            return cora_color(color_value)
        else:
            return color_value
    elif isinstance(color_value, (list, tuple, np.ndarray)):
        # RGB array
        return np.array(color_value)
    else:
        return color_value


def _apply_purpose_defaults(kwargs: Dict[str, Any], purpose: str) -> Dict[str, Any]:
    """Apply purpose-specific default values"""
    if purpose == 'fill':
        # For filled plots, ensure we have face color
        if 'facecolor' not in kwargs:
            if 'color' in kwargs:
                kwargs['facecolor'] = kwargs['color']
            else:
                kwargs['facecolor'] = default_plot_color()
        
        # Default edge color for filled plots
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'black'
    
    elif purpose == 'reachSet':
        # Reachable set specific defaults - use CORA colors
        if 'FaceColor' not in kwargs and 'facecolor' not in kwargs:
            cora_color_rgb = cora_color('CORA:reachSet')
            kwargs['FaceColor'] = cora_color_rgb  # CORA default for reachable sets
        
        # Edge color defaults to same as face color for reachable sets
        if 'EdgeColor' not in kwargs and 'edgecolor' not in kwargs:
            face_color = kwargs.get('FaceColor', kwargs.get('facecolor', cora_color('CORA:reachSet')))
            kwargs['EdgeColor'] = face_color
        
        # Ensure reachable sets are always filled and visible
        if 'alpha' not in kwargs and 'FaceAlpha' not in kwargs:
            kwargs['alpha'] = 1.0
        
        # Ensure smooth rendering
        if 'antialiased' not in kwargs:
            kwargs['antialiased'] = True
        
        # Reachable sets go in background (lowest z-order)
        if 'zorder' not in kwargs:
            kwargs['zorder'] = 1
    
    elif purpose == 'initialSet':
        # Initial set specific defaults - use CORA colors
        if 'FaceColor' not in kwargs and 'facecolor' not in kwargs:
            kwargs['FaceColor'] = cora_color('CORA:initialSet')  # CORA color for initial sets
        
        if 'EdgeColor' not in kwargs and 'edgecolor' not in kwargs:
            kwargs['EdgeColor'] = 'black'
        
        # Ensure initial set is clearly visible with high opacity
        if 'alpha' not in kwargs and 'FaceAlpha' not in kwargs:
            kwargs['alpha'] = 0.9  # High opacity for visibility
        
        # Use default MATLAB line width (no explicit linewidth = matplotlib default)
        # MATLAB doesn't specify LineWidth for initial sets, so use matplotlib default
        if 'linewidth' not in kwargs and 'LineWidth' not in kwargs:
            kwargs['linewidth'] = 1.0  # Use MATLAB default line width
        
        # High z-order for initial set visibility (above reachable sets and simulations)
        if 'zorder' not in kwargs:
            kwargs['zorder'] = 10
    
    elif purpose == 'simulation':
        # Simulation specific defaults - use CORA colors
        if 'color' not in kwargs:
            kwargs['color'] = cora_color('CORA:simulations')
        # Ensure smooth line rendering
        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = 1.0
        if 'antialiased' not in kwargs:
            kwargs['antialiased'] = True
        # Use solid lines for smooth appearance
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = '-'
        
        # Simulations go on top (highest z-order) like in MATLAB
        if 'zorder' not in kwargs:
            kwargs['zorder'] = 15  # Higher than initial set to match MATLAB behavior
    
    return kwargs 