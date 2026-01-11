"""
matZonotope package - exports matZonotope class and all its methods

This package contains the matZonotope class implementation and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .matZonotope import matZonotope
from .display import display, display_
from .dim import dim
from .numgens import numgens
from .isempty import isempty
from .size import size
from .center import center

# Attach methods to the matZonotope class
matZonotope.display = display
matZonotope.display_ = display_
matZonotope.dim = dim
matZonotope.numgens = numgens
matZonotope.isempty = isempty
matZonotope.size = size
matZonotope.center = center

# Attach display_ to __str__
matZonotope.__str__ = lambda self: display_(self)

__all__ = ['matZonotope', 'display', 'display_', 'dim', 'numgens', 'isempty', 'size', 'center'] 