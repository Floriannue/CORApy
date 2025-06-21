"""
default_plot_color - returns next color according to the colororder of the current axis

This function provides the next color in the matplotlib color cycle.

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       24-March-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def default_plot_color() -> np.ndarray:
    """
    Returns next color according to the colororder of the current axis
    
    This function mimics MATLAB's defaultPlotColor() behavior.
    
    Returns:
        RGB color triple as numpy array
    """
    try:
        # Get current axis
        ax = plt.gca()
        
        # Get the current color order
        color_order = ax._get_lines.prop_cycler.by_key()['color']
        
        # Determine color index based on whether hold is on
        # In matplotlib, we check if there are existing lines
        if hasattr(ax, '_hold') and ax._hold:
            # If hold is on, get current color index
            color_index = len(ax.lines) % len(color_order)
        else:
            # If hold is off, start from first color
            color_index = 0
        
        # Select color
        color = color_order[color_index]
        
        # Convert to RGB if it's a named color or hex
        if isinstance(color, str):
            if color.startswith('#'):
                # Hex color
                color = color[1:]
                color_rgb = tuple(int(color[i:i+2], 16)/255.0 for i in (0, 2, 4))
            else:
                # Named color - convert using matplotlib
                from matplotlib.colors import to_rgb
                color_rgb = to_rgb(color)
        else:
            color_rgb = color
            
    except Exception:
        # Fallback if no axis exists or other error
        color_rgb = [0, 0.4470, 0.7410]  # matplotlib default blue
    
    return np.array(color_rgb)
