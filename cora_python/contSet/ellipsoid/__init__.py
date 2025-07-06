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
Ellipsoid.distance = distance
Ellipsoid.volume_ = volume_
Ellipsoid.interval = interval
Ellipsoid.radius = radius
Ellipsoid.plus = plus
Ellipsoid.__add__ = plus

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
    'plus'
] 