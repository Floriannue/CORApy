"""
ConZonotope package - Constrained zonotopes

This package provides the ConZonotope class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Dmitry Grebenyuk, Mark Wetzlinger, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
"""

# Import the main ConZonotope class
from .conZonotope import ConZonotope

# Import method implementations that exist
from .dim import dim
from .isemptyobject import isemptyobject
from .empty import empty
from .origin import origin
from .generateRandom import generateRandom
from .representsa_ import representsa_

# Attach methods to the ConZonotope class
# dim and isemptyobject are required by ContSet
ConZonotope.dim = dim
ConZonotope.isemptyobject = isemptyobject
ConZonotope.representsa_ = representsa_

# Attach static methods
ConZonotope.empty = staticmethod(empty)
ConZonotope.origin = staticmethod(origin)
ConZonotope.generateRandom = staticmethod(generateRandom)

# Export the ConZonotope class and all methods
__all__ = [
    'ConZonotope',
    'dim',
    'isemptyobject',
    'empty',
    'origin',
    'generateRandom',
    'representsa_',
] 