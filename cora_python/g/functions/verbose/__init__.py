"""
Global functions - verbose module

This module contains verbose output functions including plotting utilities.
"""

from .plot import *
from .verboseLog import verboseLog, verboseLogReach, verboseLogAdaptive, verboseLogHeader, verboseLogFooter

__all__ = [
    'plot_polygon',
    'plot_polytope_3d', 
    'read_plot_options',
    'read_name_value_pair',
    'next_color',
    'default_plot_color',
    'get_unbounded_axis_limits',
    'verboseLog',
    'verboseLogReach', 
    'verboseLogAdaptive',
    'verboseLogHeader',
    'verboseLogFooter'
] 