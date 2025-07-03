"""
polygon package - exports Polygon class and all its methods

This package contains the polygon class implementation.
"""

from .polygon import Polygon
from .contains_ import contains_

# Attach methods to the class
Polygon.contains_ = contains_

# Attach static methods to the class
# Polygon.empty = staticmethod(empty)
# Polygon.generateRandom = staticmethod(generateRandom)
# Polygon.enclosePoints = staticmethod(enclosePoints)

__all__ = ['Polygon'] 