"""
ProbZonotope package - Probabilistic zonotopes

This package provides the ProbZonotope class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
"""

# Import the main ProbZonotope class
from .probZonotope import ProbZonotope

# Import method implementations that exist
from .dim import dim
from .isemptyobject import isemptyobject
from .empty import empty
from .origin import origin
from .generateRandom import generateRandom
from .sigma import sigma


ProbZonotope.dim = dim
ProbZonotope.isemptyobject = isemptyobject
ProbZonotope.sigma = sigma

# Attach static methods
ProbZonotope.empty = staticmethod(empty)
ProbZonotope.origin = staticmethod(origin)
ProbZonotope.generateRandom = staticmethod(generateRandom)

# Export the ProbZonotope class and all methods
__all__ = [
    'ProbZonotope',
    'dim',
    'isemptyobject',
    'empty',
    'origin',
    'generateRandom',
] 