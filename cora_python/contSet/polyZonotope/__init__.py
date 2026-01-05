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
from .supportFunc_ import supportFunc_
from .splitLongestGen import splitLongestGen
from .splitDepFactor import splitDepFactor
from .zonotope import zonotope
from .compact_ import compact_
from .restructure import restructure
from .approxVolumeRatio import approxVolumeRatio
from .display import display, display_

# Attach methods to the PolyZonotope class
# dim and isemptyobject are required by ContSet
PolyZonotope.dim = dim
PolyZonotope.isemptyobject = isemptyobject
PolyZonotope.interval = interval
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

# Attach display_ to __str__
PolyZonotope.__str__ = lambda self: display_(self)

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
    'supportFunc_',
    'splitLongestGen',
    'splitDepFactor',
    'zonotope',
    'compact_',
    'restructure',
    'approxVolumeRatio',
] 