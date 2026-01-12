"""
stl - Signal Temporal Logic (STL) module

This module contains the STL class and related functions for defining
and working with Signal Temporal Logic formulas.

Authors: Python translation by AI Assistant
Written: 2025
"""

from .stl import Stl
from .polytope2stl import polytope2stl
from .in_ import in_
from .finally_ import finally_

# Attach methods to the Stl class
Stl.in_ = in_
Stl.finally_ = finally_

# Alias for reserved keywords (via __getattr__ in stl.py)
# This allows x.finally() and x.in() to work as aliases for x.finally_() and x.in_()

__all__ = ['Stl', 'polytope2stl', 'in_', 'finally_'] 