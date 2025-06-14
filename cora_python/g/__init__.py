"""
Global module (g) - Mirrors cora_matlab/global

This module contains global functions and utilities used throughout CORA.
"""

from .functions import *
from .functions.matlab.validate.postprocessing.CORAerror import CORAError

__all__ = [
    'plot_polygon',
    'plot_polytope_3d', 
    'read_plot_options',
    'read_name_value_pair',
    'set_default_values',
    'input_args_check',
    'next_color',
    'default_plot_color',
    'get_unbounded_axis_limits',
    'CORAError',
]

# g module - mirrors cora_matlab/global functionality 