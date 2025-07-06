"""
specification - module for temporal logic specifications

This module contains classes and functions for defining and working with
specifications in reachability analysis, including safety, invariant, and
temporal logic specifications.

Authors: Python translation by AI Assistant
Written: 2025
"""

# Import the main specification module
from .specification import Specification

# Import stlInterval
from .stlInterval import StlInterval

__all__ = [
    'Specification', 'StlInterval'
] 