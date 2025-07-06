"""
IntervalMatrix package - Interval matrix representations

This package provides the IntervalMatrix class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

# Import the main IntervalMatrix class
from .intervalMatrix import IntervalMatrix

# Import method implementations
from .display import display
from .dim import dim
from .getPrintSetInfo import getPrintSetInfo
from .isempty import isempty

# Attach methods to the IntervalMatrix class
IntervalMatrix.display = display
IntervalMatrix.dim = dim
IntervalMatrix.getPrintSetInfo = getPrintSetInfo
IntervalMatrix.isempty = isempty

# Export the IntervalMatrix class and all methods
__all__ = [
    'IntervalMatrix',
    'display',
    'dim',
    'getPrintSetInfo',
    'isempty',
] 