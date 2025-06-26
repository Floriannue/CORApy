"""
conPolyZono package - exports ConPolyZono class and all its methods

This package contains the constrained polynomial zonotope class implementation.
"""

from .conPolyZono import ConPolyZono

# Attach static methods to the class
# ConPolyZono.empty = staticmethod(empty)
# ConPolyZono.origin = staticmethod(origin)
# ConPolyZono.generateRandom = staticmethod(generateRandom)

__all__ = ['ConPolyZono'] 