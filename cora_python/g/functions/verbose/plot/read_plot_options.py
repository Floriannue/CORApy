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
from .next_color import next_color, default_plot_color


def read_plot_options(plot_options: List[Any], purpose: str = 'none') -> Dict[str, Any]:
    """
    Read and process plot options for matplotlib
    
    Args:
        plot_options: List of LineSpec + Name-Value pairs
        purpose: Plot purpose ('fill', 'none', etc.)
        
    Returns:
        Dictionary of matplotlib plotting options
    """
    if not plot_options:
        plot_options = []
    
    # Initialize with default options
    plot_kwargs = {}
    
    # Parse linespec if provided
    linespec_kwargs = {}
    if plot_options and isinstance(plot_options[0], str):
        linespec_kwargs = _parse_linespec(plot_options[0])
        plot_options = plot_options[1:]  # Remove linespec from options
    
    # Parse name-value pairs
    nv_kwargs = _parse_name_value_pairs(plot_options)
    
    # Merge linespec and name-value pairs (name-value pairs take precedence)
    plot_kwargs.update(linespec_kwargs)
    plot_kwargs.update(nv_kwargs)
    
    # Apply purpose-specific defaults
    plot_kwargs = _apply_purpose_defaults(plot_kwargs, purpose)
    
    # Ensure we have color if not specified
    if 'color' not in plot_kwargs and 'facecolor' not in plot_kwargs:
        plot_kwargs['color'] = next_color()
    
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
        'DisplayName': 'label',
        'HandleVisibility': None  # Not directly supported in matplotlib
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
    
    elif purpose == 'initialSet':
        # Initial set specific defaults
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'black'
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = default_plot_color()
    
    return kwargs 