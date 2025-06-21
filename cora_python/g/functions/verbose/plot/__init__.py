"""
Plot utilities module

This module contains plotting functions that mirror MATLAB's CORA plotting functionality.
"""

from .plot_polygon import plot_polygon
from .plot_polytope_3d import plot_polytope_3d
from .read_plot_options import read_plot_options
from .read_name_value_pair import read_name_value_pair
from .get_unbounded_axis_limits import get_unbounded_axis_limits
from .color import cora_color, next_color, default_plot_color

__all__ = [
    'plot_polygon',
    'plot_polytope_3d', 
    'read_plot_options',
    'read_name_value_pair',
    'next_color',
    'default_plot_color',
    'get_unbounded_axis_limits',
    'cora_color'
] 