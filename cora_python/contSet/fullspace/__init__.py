"""
Fullspace package - Full-dimensional space R^n

This package provides the Fullspace class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Mark Wetzlinger, Adrian Kulmburg (MATLAB)
         Python translation by AI Assistant
"""

# Import the main Fullspace class
from .fullspace import Fullspace

# Import method implementations that exist
from .dim import dim
from .isemptyobject import isemptyobject
from .empty import empty
from .origin import origin
from .generateRandom import generateRandom
from .display import display, display_

# Import new method implementations
from .getPrintSetInfo import getPrintSetInfo
from .copy import copy
from .enclosePoints import enclosePoints
from .Inf import Inf
from .isFullDim import isFullDim
from .isBounded import isBounded
from .center import center
from .box import box
from .and_ import and_
from .not_op import not_op
from .plus import plus
from .convHull_ import convHull_
from .contains_ import contains_
from .isequal import isequal
from .isIntersecting_ import isIntersecting_
from .mtimes import mtimes
from .project import project
from .lift_ import lift_
from .projectHighDim_ import projectHighDim_
from .eventFcn import eventFcn
from .interval import interval
from .polytope import polytope
from .radius import radius
from .volume_ import volume_
from .vertices_ import vertices_
from .randPoint_ import randPoint_
from .supportFunc_ import supportFunc_
from .representsa_ import representsa_

# Attach methods to the Fullspace class
# dim and isemptyobject are required by ContSet
Fullspace.dim = dim
Fullspace.isemptyobject = isemptyobject
Fullspace.display = display
Fullspace.display_ = display_

# Attach display_ to __str__
Fullspace.__str__ = lambda self: display_(self)
Fullspace.getPrintSetInfo = getPrintSetInfo
Fullspace.copy = copy
Fullspace.isFullDim = isFullDim
Fullspace.isBounded = isBounded
Fullspace.center = center
Fullspace.box = box
Fullspace.and_ = and_
Fullspace.not_op = not_op
Fullspace.plus = plus
Fullspace.convHull_ = convHull_
Fullspace.contains_ = contains_
Fullspace.isequal = isequal
Fullspace.isIntersecting_ = isIntersecting_
Fullspace.mtimes = mtimes
Fullspace.project = project
Fullspace.lift_ = lift_
Fullspace.projectHighDim_ = projectHighDim_
Fullspace.eventFcn = eventFcn
Fullspace.interval = interval
Fullspace.polytope = polytope
Fullspace.radius = radius
Fullspace.volume_ = volume_
Fullspace.vertices_ = vertices_
Fullspace.randPoint_ = randPoint_
Fullspace.supportFunc_ = supportFunc_
Fullspace.representsa_ = representsa_

# Attach operator overloads
Fullspace.__and__ = and_
Fullspace.__invert__ = not_op
Fullspace.__add__ = plus
Fullspace.__mul__ = mtimes
Fullspace.__rmul__ = mtimes
Fullspace.__matmul__ = mtimes
Fullspace.__rmatmul__ = lambda self, other: mtimes(other, self)  # @ operator (reverse)

# Attach static methods
Fullspace.empty = staticmethod(empty)
Fullspace.origin = staticmethod(origin)
Fullspace.generateRandom = staticmethod(generateRandom)
Fullspace.enclosePoints = staticmethod(enclosePoints)
Fullspace.Inf = staticmethod(Inf)

# Export the Fullspace class and all methods
__all__ = [
    'Fullspace',
    'dim',
    'isemptyobject',
    'empty',
    'origin',
    'generateRandom',
    'display',
    'getPrintSetInfo',
    'copy',
    'enclosePoints',
    'Inf',
    'isFullDim',
    'isBounded',
    'center',
    'box',
    'and_',
    'not_op',
    'plus',
    'convHull_',
    'contains_',
    'isequal',
    'isIntersecting_',
    'mtimes',
    'project',
    'lift_',
    'projectHighDim_',
    'eventFcn',
    'interval',
    'polytope',
    'radius',
    'volume_',
    'vertices_',
    'randPoint_',
    'supportFunc_',
    'representsa_',
] 