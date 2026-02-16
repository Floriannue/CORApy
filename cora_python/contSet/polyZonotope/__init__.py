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
from .polytope import polytope
from .mtimes import mtimes
from .plus import plus
from .enclose import enclose
from .randPoint_ import randPoint_
from .interval import interval
from .center import center
from .supportFunc_ import supportFunc_
from .splitLongestGen import splitLongestGen
from .splitDepFactor import splitDepFactor
from .zonotope import zonotope
from .compact_ import compact_
from .restructure import restructure
from .approxVolumeRatio import approxVolumeRatio
from .display import display, display_
from .copy import copy
from .reduce import reduce
from .cartProd_ import cartProd_
from .quadMap import quadMap
from .linComb import linComb
from .convHull_ import convHull_
from .exactPlus import exactPlus
from .project import project
from .polygon import polygon

# Attach methods to the PolyZonotope class
# dim and isemptyobject are required by ContSet
PolyZonotope.dim = dim
PolyZonotope.isemptyobject = isemptyobject
PolyZonotope.interval = interval
PolyZonotope.center = center
PolyZonotope.representsa_ = representsa_
PolyZonotope.polytope = polytope
PolyZonotope.mtimes = mtimes
PolyZonotope.plus = plus
PolyZonotope.enclose = enclose
PolyZonotope.randPoint_ = randPoint_
PolyZonotope.supportFunc_ = supportFunc_
PolyZonotope.splitLongestGen = splitLongestGen
PolyZonotope.splitDepFactor = splitDepFactor
PolyZonotope.zonotope = zonotope
PolyZonotope.compact_ = compact_
PolyZonotope.restructure = restructure
PolyZonotope.display = display
PolyZonotope.display_ = display_
PolyZonotope.copy = copy
PolyZonotope.reduce = reduce
PolyZonotope.cartProd_ = cartProd_
PolyZonotope.quadMap = quadMap
PolyZonotope.linComb = linComb
PolyZonotope.convHull_ = convHull_
PolyZonotope.exactPlus = exactPlus
PolyZonotope.project = project
PolyZonotope.polygon = polygon

# Attach display_ to __str__
PolyZonotope.__str__ = lambda self: display_(self)

# Attach operator overloads
PolyZonotope.__matmul__ = lambda self, other: mtimes(self, other)  # @ operator (matrix multiplication)
PolyZonotope.__rmatmul__ = lambda self, other: mtimes(other, self)  # @ operator (reverse)
PolyZonotope.__mul__ = lambda self, other: mtimes(self, other)  # * operator (scalar/matrix multiplication)
PolyZonotope.__rmul__ = lambda self, other: mtimes(other, self)  # * operator (reverse, for scalar * polyZonotope)
PolyZonotope.__add__ = lambda self, other: plus(self, other)  # + operator
PolyZonotope.__radd__ = lambda self, other: plus(other, self)  # + operator (reverse)

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
    'interval',
    'center',
    'supportFunc_',
    'splitLongestGen',
    'splitDepFactor',
    'zonotope',
    'compact_',
    'restructure',
    'approxVolumeRatio',
    'copy',
    'reduce',
    'cartProd_',
    'quadMap',
    'linComb',
    'convHull_',
    'exactPlus',
    'project',
    'polygon',
] 