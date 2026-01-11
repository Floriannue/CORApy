"""
matPolytope package
"""

from .matPolytope import MatPolytope
from .display import display, display_
from .dim import dim
from .numverts import numverts
from .isempty import isempty
from .size import size

# Attach methods to the MatPolytope class
MatPolytope.display = display
MatPolytope.display_ = display_
MatPolytope.dim = dim
MatPolytope.numverts = numverts
MatPolytope.isempty = isempty
MatPolytope.size = size

# Attach display_ to __str__
MatPolytope.__str__ = lambda self: display_(self)

__all__ = [
    'MatPolytope',
    'display',
    'display_',
    'dim',
    'numverts',
    'isempty',
    'size',
]
