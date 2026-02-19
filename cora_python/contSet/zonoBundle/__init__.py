"""
ZonoBundle package - Zonotope bundles

This package provides the ZonoBundle class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

# Import the main ZonoBundle class
from .zonoBundle import ZonoBundle

# Import method implementations that exist
from .dim import dim
from .isemptyobject import isemptyobject
from .empty import empty
from .origin import origin
from .generateRandom import generateRandom
from .display import display, display_
from .interval import interval
from .center import center
from .conZonotope import conZonotope
from .representsa_ import representsa_
from .and_ import and_
from .split import split
from .plus import plus
from .copy import copy
from .mtimes import mtimes
from .enclose import enclose
from .reduce import reduce
from .cartProd_ import cartProd_
from .quadMap import quadMap
from .project import project
from .polytope import polytope
from .vertices_ import vertices_
from .supportFunc_ import supportFunc_
from .randPoint_ import randPoint_

# Attach methods to the ZonoBundle class
# dim and isemptyobject are required by ContSet
ZonoBundle.dim = dim
ZonoBundle.isemptyobject = isemptyobject
ZonoBundle.display = display
ZonoBundle.display_ = display_

# Attach display_ to __str__
ZonoBundle.__str__ = lambda self: display_(self)
ZonoBundle.interval = interval
ZonoBundle.center = center
ZonoBundle.conZonotope = conZonotope
ZonoBundle.representsa_ = representsa_
ZonoBundle.and_ = and_
ZonoBundle.__and__ = and_
ZonoBundle.split = split
ZonoBundle.plus = plus
ZonoBundle.__add__ = plus
ZonoBundle.copy = copy
ZonoBundle.mtimes = mtimes
ZonoBundle.__mul__ = mtimes
ZonoBundle.__rmul__ = lambda self, other: mtimes(other, self)
ZonoBundle.__matmul__ = mtimes
ZonoBundle.__rmatmul__ = lambda self, other: mtimes(other, self)
ZonoBundle.enclose = enclose
ZonoBundle.reduce = reduce
ZonoBundle.cartProd_ = cartProd_
ZonoBundle.quadMap = quadMap
ZonoBundle.project = project
ZonoBundle.polytope = polytope
ZonoBundle.vertices_ = vertices_
ZonoBundle.supportFunc_ = supportFunc_
ZonoBundle.randPoint_ = randPoint_

# Attach static methods
ZonoBundle.empty = staticmethod(empty)
ZonoBundle.origin = staticmethod(origin)
ZonoBundle.generateRandom = staticmethod(generateRandom)

# Export the ZonoBundle class and all methods
__all__ = [
    'ZonoBundle',
    'dim',
    'isemptyobject',
    'display',
    'interval',
    'center',
    'conZonotope',
    'representsa_',
    'and_',
    'split',
    'plus',
    'copy',
    'mtimes',
    'enclose',
    'reduce',
    'cartProd_',
    'quadMap',
    'project',
    'polytope',
    'vertices_',
    'supportFunc_',
    'randPoint_',
    'empty',
    'origin',
    'generateRandom',
] 