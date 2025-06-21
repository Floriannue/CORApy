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
    Short-hand for cora_color('CORA:next')
    
    This function mimics MATLAB's nextcolor() function.
    
    Returns:
        RGB color array
    """
    from .cora_color import cora_color
    return cora_color('CORA:next')


def nextcolor() -> np.ndarray:
    """
    Alias for next_color() for MATLAB compatibility
    
    Returns:
        RGB color array
    """
    return next_color()
