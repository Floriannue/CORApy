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
from .display import display, display_
from .dim import dim
from .getPrintSetInfo import getPrintSetInfo
from .isempty import isempty
from .mtimes import mtimes
from .abs import abs
from .powers import powers
from .dependentTerms import dependentTerms
from .exponentialRemainder import exponentialRemainder
from .expm import expm
from .expmInd import expmInd

# Attach methods to the IntervalMatrix class
IntervalMatrix.display = display
IntervalMatrix.display_ = display_

# Attach display_ to __str__
IntervalMatrix.__str__ = lambda self: display_(self)
IntervalMatrix.dim = dim
IntervalMatrix.getPrintSetInfo = getPrintSetInfo
IntervalMatrix.isempty = isempty
IntervalMatrix.abs = abs

# Export the IntervalMatrix class and all methods
__all__ = [
    'IntervalMatrix',
    'display',
    'dim',
    'getPrintSetInfo',
    'isempty',
    'mtimes',
    'abs',
    'powers',
    'dependentTerms',
    'exponentialRemainder',
    'expm',
    'expmInd',
] 