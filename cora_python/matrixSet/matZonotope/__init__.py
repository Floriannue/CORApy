"""
matZonotope package - exports matZonotope class and all its methods

This package contains the matZonotope class implementation and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .matZonotope import matZonotope
from .display import display, display_
from .dim import dim
from .numgens import numgens
from .isempty import isempty
from .size import size
from .center import center
from .mtimes import mtimes
from .dependentTerms import dependentTerms
from .powers import powers
from .expmOneParam import expmOneParam
from .expmMixed import expmMixed
from .expmIndMixed import expmIndMixed
from .mpower import mpower

# Attach methods to the matZonotope class
matZonotope.display = display
matZonotope.display_ = display_
matZonotope.dim = dim
matZonotope.numgens = numgens
matZonotope.isempty = isempty
matZonotope.size = size
matZonotope.center = center
matZonotope.mpower = mpower

# Attach operator overloads
matZonotope.__mul__ = lambda self, other: mtimes(self, other)
matZonotope.__rmul__ = lambda self, other: mtimes(other, self)
matZonotope.__pow__ = lambda self, other: mpower(self, other)
matZonotope.__str__ = lambda self: display_(self)
matZonotope.__repr__ = lambda self: display_(self)

__all__ = ['matZonotope', 'display', 'display_', 'dim', 'numgens', 'isempty', 'size', 'center',
           'mtimes', 'mpower', 'dependentTerms', 'powers', 'expmOneParam', 'expmMixed', 'expmIndMixed'] 