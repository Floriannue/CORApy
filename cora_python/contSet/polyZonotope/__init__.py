"""
PolyZonotope package - Polynomial zonotopes

This package provides the PolyZonotope class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Niklas Kochdumper, Mark Wetzlinger, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
"""

# Import the main PolyZonotope class
from .polyZonotope import PolyZonotope

# Import method implementations that exist
from .dim import dim
from .isemptyobject import isemptyobject
from .empty import empty
from .origin import origin
from .generateRandom import generateRandom
from .representsa_ import representsa_

# Attach methods to the PolyZonotope class
# dim and isemptyobject are required by ContSet
PolyZonotope.dim = dim
PolyZonotope.isemptyobject = isemptyobject
PolyZonotope.representsa_ = representsa_

# Attach static methods
PolyZonotope.empty = staticmethod(empty)
PolyZonotope.origin = staticmethod(origin)
PolyZonotope.generateRandom = staticmethod(generateRandom)

# Export the PolyZonotope class and all methods
__all__ = [
    'PolyZonotope',
    'dim',
    'isemptyobject',
    'empty',
    'origin',
    'generateRandom',
    'representsa_',
] 