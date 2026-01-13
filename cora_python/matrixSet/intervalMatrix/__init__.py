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
from .plus import plus
from .center import center
from .rad import rad
from .delta import delta
from .shape import shape
from .mpower import mpower
from .powers import powers
from .dependentTerms import dependentTerms
from .exponentialRemainder import exponentialRemainder
from .expm import expm
from .expmInd import expmInd

# Attach methods to the IntervalMatrix class
IntervalMatrix.display = display
IntervalMatrix.display_ = display_

# display_ is attached as __str__ below
IntervalMatrix.dim = dim
IntervalMatrix.getPrintSetInfo = getPrintSetInfo
IntervalMatrix.isempty = isempty
IntervalMatrix.abs = abs
IntervalMatrix.plus = plus
IntervalMatrix.center = center
IntervalMatrix.rad = rad
IntervalMatrix.delta = delta
IntervalMatrix.shape = shape
IntervalMatrix.mpower = mpower
IntervalMatrix.__add__ = lambda self, other: plus(self, other)
IntervalMatrix.__radd__ = lambda self, other: plus(other, self)
IntervalMatrix.__mul__ = lambda self, other: mtimes(self, other)
IntervalMatrix.__rmul__ = lambda self, other: mtimes(other, self)
IntervalMatrix.__pow__ = lambda self, other: mpower(self, other)
IntervalMatrix.__repr__ = lambda self: display_(self)
IntervalMatrix.__str__ = lambda self: display_(self)

# Export the IntervalMatrix class and all methods
__all__ = [
    'IntervalMatrix',
    'display',
    'dim',
    'getPrintSetInfo',
    'isempty',
    'mtimes',
    'abs',
    'plus',
    'center',
    'rad',
    'delta',
    'shape',
    'mpower',
    'powers',
    'dependentTerms',
    'exponentialRemainder',
    'expm',
    'expmInd',
] 