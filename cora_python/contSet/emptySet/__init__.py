# Import the main class
from .emptySet import EmptySet

# Import methods that exist as separate .m files in MATLAB
from .mtimes import mtimes
from .plus import plus
from .and_ import and_
from .center import center
from .dim import dim
from .display import display
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

# Attach methods to the class
EmptySet.mtimes = mtimes
EmptySet.plus = plus
EmptySet.and_ = and_
EmptySet.center = center
EmptySet.dim = dim
EmptySet.display = display
EmptySet.isemptyobject = isemptyobject
EmptySet.copy = copy
EmptySet.isBounded = isBounded
EmptySet.supportFunc_ = supportFunc_
EmptySet.representsa_ = representsa_
EmptySet.contains_ = contains_
EmptySet.isequal = isequal
EmptySet.volume_ = volume_

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
    'volume_'
] 