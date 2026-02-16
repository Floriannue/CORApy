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
from .center import center
from .sigma import sigma
from .display import display, display_


ProbZonotope.dim = dim
ProbZonotope.display = display
ProbZonotope.display_ = display_

# Attach display_ to __str__
ProbZonotope.__str__ = lambda self: display_(self)
ProbZonotope.isemptyobject = isemptyobject
ProbZonotope.sigma = sigma
ProbZonotope.center = center

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
    'center',
] 