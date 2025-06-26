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
from .dim import dim
from .center import center
from .contains_ import contains_

# Attach methods to the class (dim and is_empty implemented directly in class)
Capsule.empty = staticmethod(empty)
Capsule.origin = staticmethod(origin)
Capsule.display = display
Capsule.representsa_ = representsa_
Capsule.is_empty = isemptyobject
Capsule.isemptyobject = isemptyobject
Capsule.dim = dim
Capsule.center = center
Capsule.contains_ = contains_

__all__ = ['Capsule', 'empty', 'origin', 'display', 'representsa_', 'isemptyobject', 'dim', 'center', 'contains_'] 