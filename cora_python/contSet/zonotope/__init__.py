"""
Zonotope package - exports zonotope class and all its methods

This package contains the zonotope class implementation and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .zonotope import Zonotope
from .abs import abs_
from .box import box
from .plus import plus
from .mtimes import mtimes
from .dim import dim
from .empty import empty
from .origin import origin
from .isemptyobject import isemptyobject
from .display import display
from .randPoint_ import randPoint_
from .vertices_ import vertices_
from .project import project
from .center import center
from .representsa_ import representsa_
from .compact_ import compact_
from .interval import interval
from .contains_ import contains_
from .norm_ import norm_
from .zonotopeNorm import zonotopeNorm

# Attach methods to the class
Zonotope.abs_ = abs_
Zonotope.box = box
Zonotope.plus = plus
Zonotope.mtimes = mtimes
Zonotope.dim = dim
Zonotope.empty = empty
Zonotope.origin = origin
Zonotope.isemptyobject = isemptyobject
Zonotope.display = display
Zonotope.randPoint_ = randPoint_
Zonotope.vertices_ = vertices_
Zonotope.project = project
Zonotope.center = center
Zonotope.representsa_ = representsa_
Zonotope.compact_ = compact_
Zonotope.interval = interval
Zonotope.contains_ = contains_
Zonotope.norm_ = norm_
Zonotope.zonotopeNorm = zonotopeNorm

# Special methods
Zonotope.__abs__ = abs_
Zonotope.__add__ = plus
Zonotope.__mul__ = mtimes
Zonotope.__rmul__ = lambda self, other: mtimes(other, self)

__all__ = ['Zonotope', 'abs_', 'box', 'plus', 'mtimes', 'dim', 'empty', 'origin', 'isemptyobject', 'display', 'randPoint_', 'vertices_', 'project', 'center', 'representsa_', 'compact_', 'interval', 'contains_', 'norm_', 'zonotopeNorm'] 