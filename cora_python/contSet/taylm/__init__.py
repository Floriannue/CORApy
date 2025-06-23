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

# Attach methods to the Taylm class
# dim and isemptyobject are required by parent class (but Taylm doesn't inherit from ContSet)
Taylm.dim = dim
Taylm.isemptyobject = isemptyobject

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
] 