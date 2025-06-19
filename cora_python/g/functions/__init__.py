"""
Global functions module

This module contains utility functions used across the CORA Python library.
"""

from .verbose import *
from .matlab import *

__all__ = [
    'plot_polygon',
    'plot_polytope_3d', 
    'read_plot_options',
    'read_name_value_pair',
    'next_color',
    'default_plot_color',
    'get_unbounded_axis_limits',
] 