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
from .display import display

# Attach methods to the Fullspace class
# dim and isemptyobject are required by ContSet
Fullspace.dim = dim
Fullspace.isemptyobject = isemptyobject
Fullspace.display = display

# Attach static methods
Fullspace.empty = staticmethod(empty)
Fullspace.origin = staticmethod(origin)
Fullspace.generateRandom = staticmethod(generateRandom)

# Export the Fullspace class and all methods
__all__ = [
    'Fullspace',
    'dim',
    'isemptyobject',
    'empty',
    'origin',
    'generateRandom',
    'display',
] 