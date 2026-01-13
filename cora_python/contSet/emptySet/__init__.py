# Import the main class
from .emptySet import EmptySet

# Import methods that exist as separate .m files in MATLAB
from .mtimes import mtimes
from .plus import plus
from .and_ import and_
from .center import center
from .dim import dim
from .display import display, display_
from .isemptyobject import isemptyobject
from .empty import empty
from .generateRandom import generateRandom
from .copy import copy
from .isBounded import isBounded
from .supportFunc_ import supportFunc_
from .representsa_ import representsa_
from .contains_ import contains_
from .isequal import isequal
from .volume_ import volume_
from .convHull_ import convHull_
from .getPrintSetInfo import getPrintSetInfo
from .interval import interval
from .isFullDim import isFullDim
from .isIntersecting_ import isIntersecting_
from .lift_ import lift_
from .not_op import not_op
from .polytope import polytope
from .project import project
from .projectHighDim_ import projectHighDim_
from .radius import radius
from .randPoint_ import randPoint_
from .vertices_ import vertices_

# Attach methods to the class
EmptySet.mtimes = mtimes
EmptySet.plus = plus
EmptySet.and_ = and_
EmptySet.center = center
EmptySet.dim = dim
EmptySet.display = display
EmptySet.display_ = display_

# Attach display_ to __str__
EmptySet.__str__ = lambda self: display_(self)
EmptySet.isemptyobject = isemptyobject
EmptySet.copy = copy
EmptySet.isBounded = isBounded
EmptySet.supportFunc_ = supportFunc_
EmptySet.representsa_ = representsa_
EmptySet.contains_ = contains_
EmptySet.isequal = isequal
EmptySet.volume_ = volume_
EmptySet.convHull_ = convHull_
EmptySet.getPrintSetInfo = getPrintSetInfo
EmptySet.interval = interval
EmptySet.isFullDim = isFullDim
EmptySet.isIntersecting_ = isIntersecting_
EmptySet.lift_ = lift_
EmptySet.not_op = not_op
EmptySet.polytope = polytope
EmptySet.project = project
EmptySet.projectHighDim_ = projectHighDim_
EmptySet.radius = radius
EmptySet.randPoint_ = randPoint_
EmptySet.vertices_ = vertices_

# Overload operators
EmptySet.__invert__ = not_op  # ~ operator
EmptySet.__add__ = plus  # + operator
EmptySet.__radd__ = plus  # + operator (right side)
EmptySet.__matmul__ = mtimes  # @ operator (matrix multiplication)
EmptySet.__rmatmul__ = lambda self, other: mtimes(other, self)  # @ operator (reverse)

# Attach static methods
EmptySet.empty = staticmethod(empty)
EmptySet.generateRandom = staticmethod(generateRandom)

# Define what gets exported when using "from emptySet import *"
__all__ = [
    'EmptySet',
    'mtimes',
    'plus', 
    'and_',
    'center',
    'dim',
    'display',
    'isemptyobject',
    'empty',
    'generateRandom',
    'copy',
    'isBounded',
    'supportFunc_',
    'representsa_',
    'contains_',
    'isequal',
    'volume_',
    'convHull_',
    'getPrintSetInfo',
    'interval',
    'isFullDim',
    'isIntersecting_',
    'lift_',
    'not_op',
    'polytope',
    'project',
    'projectHighDim_',
    'radius',
    'randPoint_',
    'vertices_'
] 