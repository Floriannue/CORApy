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

# Attach methods to the class
EmptySet.mtimes = mtimes
EmptySet.plus = plus
EmptySet.and_ = and_
EmptySet.center = center
EmptySet.dim = dim
EmptySet.display = display
EmptySet.isemptyobject = isemptyobject

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
    'generateRandom'
] 