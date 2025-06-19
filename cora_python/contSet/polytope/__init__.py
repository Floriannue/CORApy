"""
Polytope module for CORA

This module contains the Polytope class for representing
convex polytopes defined by linear constraints.
"""

# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

from .polytope import Polytope
from .zonotope import zonotope
from .contains_ import contains_
from .dim import dim
from .center import center
from .isBounded import isBounded
from .isemptyobject import isemptyobject
from .interval import interval
from .empty import empty
from .Inf import Inf
from .origin import origin
from .copy import copy
from .display import display
from .mtimes import mtimes
from .plus import plus
from .project import project
from .representsa_ import representsa_
from .constraints import constraints
from .vertices_ import vertices_
from .minus import minus, rminus


Polytope.zonotope = zonotope
Polytope.contains_ = contains_
Polytope.dim = dim
Polytope.center = center
Polytope.isBounded = isBounded
Polytope.isemptyobject = isemptyobject
Polytope.interval = interval
Polytope.empty = empty
Polytope.Inf = Inf
Polytope.origin = origin
Polytope.copy = copy
Polytope.display = display
Polytope.mtimes = mtimes
Polytope.plus = plus
Polytope.project = project
Polytope.representsa_ = representsa_
Polytope.constraints = constraints
Polytope.vertices_ = vertices_
Polytope.__sub__ = minus
Polytope.__rsub__ = rminus
Polytope.__mul__ = mtimes
Polytope.__rmul__ = mtimes

__all__ = [
    'Polytope',
    'dim',
    'empty',
    'Inf',
    'origin',
    'copy',
    'display',
    'mtimes',
    'plus',
    'project',
    'representsa_',
    'constraints',
] 