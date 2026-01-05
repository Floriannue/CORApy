"""
Taylm package - Taylor models

This package provides the Taylm class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Dmitry Grebenyuk, Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

# Import the main Taylm class
from .taylm import Taylm

# Import method implementations that exist
from .dim import dim
from .isemptyobject import isemptyobject
from .empty import empty
from .origin import origin
from .generateRandom import generateRandom
from .mtimes import mtimes
from .plus import plus
from .times import times
from .display import display, display_

# Attach methods to the Taylm class
# dim and isemptyobject are required by parent class (but Taylm doesn't inherit from ContSet)
Taylm.dim = dim
Taylm.isemptyobject = isemptyobject

# Attach arithmetic methods
Taylm.mtimes = mtimes
Taylm.plus = plus
Taylm.times = times

# Attach magic methods for Python operators using lambdas
Taylm.__matmul__ = lambda self, other: mtimes(other, self)
Taylm.__rmatmul__ = lambda self, other: mtimes(self, other)
Taylm.__add__ = lambda self, other: plus(self, other)
Taylm.__radd__ = lambda self, other: plus(other, self)
Taylm.__mul__ = lambda self, other: times(self, other)
Taylm.__rmul__ = lambda self, other: times(other, self)
Taylm.display = display
Taylm.display_ = display_

# Attach display_ to __str__
Taylm.__str__ = lambda self: display_(self)

# Attach static methods
Taylm.empty = staticmethod(empty)
Taylm.origin = staticmethod(origin)
Taylm.generateRandom = staticmethod(generateRandom)

# Export the Taylm class and all methods
__all__ = [
    'Taylm',
    'dim',
    'isemptyobject',
    'empty',
    'origin',
    'generateRandom',
    'mtimes',
    'plus',
    'times',
] 