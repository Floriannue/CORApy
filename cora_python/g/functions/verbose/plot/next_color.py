"""
next_color and default_plot_color - color management utilities

These functions provide color management for plotting, mimicking MATLAB's
color order functionality.

Syntax:
    color = next_color()
    color = default_plot_color()

Outputs:
    color - RGB color array or color string

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 2023 (MATLAB)
Python translation: 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def next_color() -> np.ndarray:
    """
    Get the next color from the current matplotlib color cycle
    
    Returns:
        RGB color array
    """
    # Use the default matplotlib color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Try to get current axes to determine color index
    try:
        ax = plt.gca()
        # Count existing lines to determine next color index
        num_lines = len(ax.lines)
        color_index = num_lines % len(colors)
        color = colors[color_index]
    except:
        # Fallback to first color if no axes available
        color = colors[0] if colors else 'blue'
    
    # Convert to RGB if it's a named color
    if isinstance(color, str):
        if color.startswith('#'):
            # Hex color
            color = color[1:]
            rgb = tuple(int(color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        else:
            # Named color - convert using matplotlib
            from matplotlib.colors import to_rgb
            rgb = to_rgb(color)
    else:
        rgb = color
    
    return np.array(rgb)


def default_plot_color() -> np.ndarray:
    """
    Get the default plot color (usually the first color in the cycle)
    
    Returns:
        RGB color array
    """
    # Get default color from matplotlib
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    default_color = colors[0] if colors else 'blue'
    
    # Convert to RGB
    if isinstance(default_color, str):
        if default_color.startswith('#'):
            # Hex color
            color = default_color[1:]
            rgb = tuple(int(color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        else:
            # Named color
            from matplotlib.colors import to_rgb
            rgb = to_rgb(default_color)
    else:
        rgb = default_color
    
    return np.array(rgb) 