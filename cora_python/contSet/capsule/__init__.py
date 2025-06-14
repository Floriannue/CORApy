"""
Capsule package - exports capsule class and all its methods

This package contains the capsule class implementation and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .capsule import Capsule
from .empty import empty
from .origin import origin
from .display import display
from .representsa_ import representsa_
from .isemptyobject import isemptyobject

# Attach static methods to the class
Capsule.empty = empty
Capsule.origin = origin
Capsule.display = display

__all__ = ['Capsule', 'empty', 'origin', 'display'] 