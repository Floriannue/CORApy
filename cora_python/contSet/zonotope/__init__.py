"""
Zonotope package - exports zonotope class and all its methods

This package contains the zonotope class implementation and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .zonotope import Zonotope
from .plus import plus
from .mtimes import mtimes
from .dim import dim
from .empty import empty
from .origin import origin
from .isemptyobject import isemptyobject
from .display import display
from .randPoint import randPoint

# Attach methods to the class
Zonotope.plus = plus
Zonotope.mtimes = mtimes
Zonotope.dim = dim
Zonotope.empty = empty
Zonotope.origin = origin
Zonotope.isemptyobject = isemptyobject
Zonotope.display = display
Zonotope.randPoint = randPoint

__all__ = ['Zonotope', 'plus', 'mtimes', 'dim', 'empty', 'origin', 'isemptyobject', 'display', 'randPoint'] 