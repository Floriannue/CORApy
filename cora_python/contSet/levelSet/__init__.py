"""
levelSet package - exports LevelSet class and all its methods

This package contains the level set class implementation.
"""

from .levelSet import LevelSet
from .empty import empty
from .dim import dim

# Attach static methods to the class
LevelSet.empty = staticmethod(empty)

# Attach instance methods to the class
LevelSet.dim = dim

# LevelSet.generateRandom = staticmethod(generateRandom)
# LevelSet.Inf = staticmethod(Inf)

__all__ = ['LevelSet'] 