"""
conPolyZono package - exports ConPolyZono class and all its methods

This package contains the constrained polynomial zonotope class implementation.
"""

from .conPolyZono import ConPolyZono
from .dim import dim
from .isemptyobject import isemptyobject
from .empty import empty
from .representsa_ import representsa_
from .display import display, display_

# Attach static methods to the class
ConPolyZono.empty = staticmethod(empty)
# ConPolyZono.origin = staticmethod(origin)
# ConPolyZono.generateRandom = staticmethod(generateRandom)

ConPolyZono.dim = dim
ConPolyZono.isemptyobject = isemptyobject
ConPolyZono.representsa_ = representsa_
ConPolyZono.display = display
ConPolyZono.display_ = display_

# Attach display_ to __str__
ConPolyZono.__str__ = lambda self: display_(self)

__all__ = ['ConPolyZono'] 