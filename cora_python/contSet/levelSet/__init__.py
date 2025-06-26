"""
levelSet package - exports LevelSet class and all its methods

This package contains the level set class implementation.
"""

from .levelSet import LevelSet

# Attach static methods to the class
# LevelSet.empty = staticmethod(empty)
# LevelSet.generateRandom = staticmethod(generateRandom)
# LevelSet.Inf = staticmethod(Inf)

__all__ = ['LevelSet'] 