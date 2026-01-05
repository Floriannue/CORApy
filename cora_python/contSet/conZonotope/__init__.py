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
from .minkDiff import minkDiff
from .vertices_ import vertices_
from .compact import compact
from .reduceConstraints import reduceConstraints
from .plus import plus
from .uminus import uminus
from .center import center
from .display import display, display_

# Attach methods to the ConZonotope class
# dim and isemptyobject are required by ContSet
ConZonotope.center = center
ConZonotope.dim = dim
ConZonotope.isemptyobject = isemptyobject
ConZonotope.representsa_ = representsa_
ConZonotope.minkDiff = minkDiff
ConZonotope.vertices_ = vertices_
ConZonotope.compact = compact
ConZonotope.reduceConstraints = reduceConstraints
ConZonotope.plus = plus
ConZonotope.__add__ = plus
ConZonotope.uminus = uminus
ConZonotope.__neg__ = uminus
ConZonotope.display = display
ConZonotope.display_ = display_

# Attach display_ to __str__
ConZonotope.__str__ = lambda self: display_(self)

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
    'minkDiff',
    'vertices_',
    'compact',
    'reduceConstraints',
    'plus',
    'uminus',
] 