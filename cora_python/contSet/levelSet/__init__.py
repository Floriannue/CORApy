"""
levelSet package - exports LevelSet class and all its methods

This package contains the level set class implementation.
"""

from .levelSet import LevelSet
from .empty import empty
from .dim import dim
from .display import display, display_

# Attach static methods to the class
LevelSet.empty = staticmethod(empty)

# Attach instance methods to the class
LevelSet.dim = dim
LevelSet.display = display
LevelSet.display_ = display_

# Attach display_ to __str__
LevelSet.__str__ = lambda self: display_(self)

# LevelSet.generateRandom = staticmethod(generateRandom)
# LevelSet.Inf = staticmethod(Inf)

__all__ = ['LevelSet'] 