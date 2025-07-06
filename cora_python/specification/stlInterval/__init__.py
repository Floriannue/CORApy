"""
StlInterval package - Signal Temporal Logic intervals

This package provides the StlInterval class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Florian Lercher (MATLAB)
         Python translation by AI Assistant
"""

# Import the main StlInterval class
from .stlInterval import StlInterval

# Import method implementations
from .dim import dim
from .isemptyobject import isemptyobject
from .isequal import isequal
from .empty import empty

# Attach methods to the StlInterval class
StlInterval.dim = dim
StlInterval.isemptyobject = isemptyobject
StlInterval.isequal = isequal

# Attach operator overloading
StlInterval.__eq__ = lambda self, other: isequal(self, other)
StlInterval.__ne__ = lambda self, other: not isequal(self, other)

# Attach static methods
StlInterval.empty = staticmethod(empty)

# Export the StlInterval class and methods
__all__ = [
    'StlInterval',
    'dim',
    'isemptyobject', 
    'isequal',
    'empty'
] 