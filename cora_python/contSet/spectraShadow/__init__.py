"""
SpectraShadow package - Spectrahedral shadows

This package provides the SpectraShadow class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Adrian Kulmburg (MATLAB)
         Python translation by AI Assistant
"""

# Import the main SpectraShadow class
from .spectraShadow import SpectraShadow

# Import method implementations that exist
from .dim import dim
from .isemptyobject import isemptyobject
from .empty import empty
from .origin import origin
from .generateRandom import generateRandom

# Attach methods to the SpectraShadow class
# dim and isemptyobject are required by ContSet
SpectraShadow.dim = dim
SpectraShadow.isemptyobject = isemptyobject

# Attach static methods
SpectraShadow.empty = staticmethod(empty)
SpectraShadow.origin = staticmethod(origin)
SpectraShadow.generateRandom = staticmethod(generateRandom)

# Export the SpectraShadow class and all methods
__all__ = [
    'SpectraShadow',
    'dim',
    'isemptyobject',
    'empty',
    'origin',
    'generateRandom',
] 