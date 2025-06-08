"""
Zonotope package - exports zonotope class and all its methods

This package contains the zonotope class implementation and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .zonotope import zonotope
from .plus import plus
from .mtimes import mtimes
from .dim import dim
from .empty import empty
from .origin import origin
from .isemptyobject import isemptyobject
from .display import display

# Attach methods to the class
zonotope.plus = plus
zonotope.mtimes = mtimes
zonotope.dim = dim
zonotope.empty = empty
zonotope.origin = origin
zonotope.isemptyobject = isemptyobject
zonotope.display = display

__all__ = ['zonotope', 'plus', 'mtimes', 'dim', 'empty', 'origin', 'isemptyobject', 'display'] 