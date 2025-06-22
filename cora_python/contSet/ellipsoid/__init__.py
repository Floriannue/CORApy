from .ellipsoid import Ellipsoid
from .dim import dim
from .isemptyobject import isemptyobject
from .center import center
from .display import display
from .empty import empty
from .representsa_ import representsa_
from .contains_ import contains_
from .ellipsoidNorm import ellipsoidNorm
from .isFullDim import isFullDim
from .rank import rank
from .generators import generators
from .zonotope import zonotope

# Attach methods to the Ellipsoid class
Ellipsoid.dim = dim
Ellipsoid.isemptyobject = isemptyobject
Ellipsoid.center = center
Ellipsoid.display = display
Ellipsoid.empty = staticmethod(empty)
Ellipsoid.is_empty = isemptyobject
Ellipsoid.representsa_ = representsa_
Ellipsoid.contains_ = contains_
Ellipsoid.ellipsoidNorm = ellipsoidNorm
Ellipsoid.isFullDim = isFullDim
Ellipsoid.rank = rank
Ellipsoid.generators = generators
Ellipsoid.zonotope = zonotope

__all__ = [
    'Ellipsoid',
    'dim',
    'isemptyobject',
    'center',
    'display',
    'empty',
    'representsa_',
    'contains_',
    'ellipsoidNorm',
    'isFullDim',
    'rank',
    'generators',
    'zonotope'
] 