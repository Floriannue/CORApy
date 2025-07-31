"""
This module exports the Ellipsoid class and attaches its methods.
"""

from .ellipsoid import Ellipsoid
from .center import center
from .contains_ import contains_
from .dim import dim
from .display import display
from .ellipsoidNorm import ellipsoidNorm
from .empty import empty
from .enclosePoints import enclosePoints
from .generators import generators
from .isemptyobject import isemptyobject
from .isFullDim import isFullDim
from .rank import rank
from .representsa_ import representsa_
from .supportFunc_ import supportFunc_
from .zonotope import zonotope
from .mtimes import mtimes
from .copy import copy
from .project import project
from .distance import distance
from .volume_ import volume_
from .interval import interval
from .radius import radius
from .plus import plus
from .origin import origin
from .vertices_ import vertices_
from .norm_ import norm_
from .or_ import or_
from .randPoint_ import randPoint_
from .isBounded import isBounded
from .isnan import isnan
from .getPrintSetInfo import getPrintSetInfo
from .isBadDir import isBadDir
from .isequal import isequal
from .reduce import reduce

# Attach methods to the Ellipsoid class
Ellipsoid.center = center
Ellipsoid.contains_ = contains_
Ellipsoid.dim = dim
Ellipsoid.display = display
Ellipsoid.ellipsoidNorm = ellipsoidNorm
Ellipsoid.empty = staticmethod(empty)
Ellipsoid.enclosePoints = staticmethod(enclosePoints)
Ellipsoid.generators = generators
Ellipsoid.isemptyobject = isemptyobject
Ellipsoid.isFullDim = isFullDim
Ellipsoid.rank = rank
Ellipsoid.representsa_ = representsa_
Ellipsoid.supportFunc_ = supportFunc_
Ellipsoid.zonotope = zonotope
Ellipsoid.copy = copy
Ellipsoid.project = project
Ellipsoid.mtimes = mtimes
Ellipsoid.__matmul__ = lambda self, other: mtimes(self, other)
Ellipsoid.__rmatmul__ = lambda self, other: mtimes(other, self)
Ellipsoid.__mul__ = lambda self, other: mtimes(self, other) # For scalar multiplication
Ellipsoid.__rmul__ = lambda self, other: mtimes(other, self) # For scalar multiplication
Ellipsoid.distance = distance
Ellipsoid.volume_ = volume_
Ellipsoid.interval = interval
Ellipsoid.radius = radius
Ellipsoid.plus = plus
Ellipsoid.__add__ = plus
Ellipsoid.__radd__ = plus
Ellipsoid.origin = origin
Ellipsoid.vertices_ = vertices_
Ellipsoid.norm_ = norm_
Ellipsoid.or_ = or_
Ellipsoid.randPoint_ = randPoint_
Ellipsoid.isBounded = isBounded
Ellipsoid.isnan = isnan
Ellipsoid.getPrintSetInfo = getPrintSetInfo
Ellipsoid.isBadDir = isBadDir
Ellipsoid.isequal = isequal
Ellipsoid.reduce = reduce


__all__ = [
    'Ellipsoid',
    'center',
    'contains_',
    'dim',
    'display',
    'ellipsoidNorm',
    'empty',
    'enclosePoints',
    'generators',
    'isemptyobject',
    'isFullDim',
    'rank',
    'representsa_',
    'supportFunc_',
    'zonotope',
    'mtimes',
    'copy',
    'project',
    'distance',
    'volume_',
    'interval',
    'radius',
    'plus',
    'origin',
    'vertices_',
    'norm_',
    'or_',
    'randPoint_',
    'isBounded',
    'isnan',
    'getPrintSetInfo',
    'isBadDir',
    'isequal',
    'reduce'
] 