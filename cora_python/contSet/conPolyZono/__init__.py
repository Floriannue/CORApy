"""
conPolyZono package - exports ConPolyZono class and all its methods

This package contains the constrained polynomial zonotope class implementation.
"""

from .conPolyZono import ConPolyZono
from .dim import dim
from .isemptyobject import isemptyobject

# Attach static methods to the class
# ConPolyZono.empty = staticmethod(empty)
# ConPolyZono.origin = staticmethod(origin)
# ConPolyZono.generateRandom = staticmethod(generateRandom)

ConPolyZono.dim = dim
ConPolyZono.isemptyobject = isemptyobject

__all__ = ['ConPolyZono'] 