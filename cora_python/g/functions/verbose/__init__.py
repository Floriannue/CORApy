"""
Global functions - verbose module

This module contains verbose output functions including plotting utilities.
"""

from .plot import *

__all__ = [
    'plot_polygon',
    'plot_polytope_3d', 
    'read_plot_options',
    'read_name_value_pair',
    'set_default_values',
    'input_args_check',
    'next_color',
    'default_plot_color',
    'get_unbounded_axis_limits'
] 