"""
Polytope module for CORA

This module contains the Polytope class for representing
convex polytopes defined by linear constraints.
"""

# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

from .polytope import Polytope

# Import all method functions
from .zonotope import zonotope
from .contains_ import contains_
from .center import center
from .dim import dim
from .ellipsoid import ellipsoid
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
from .generate_random import generate_random
from .get_print_set_info import get_print_set_info
from .enclose_points import enclose_points
from .isIntersecting_ import isIntersecting_
from .normalizeConstraints import normalizeConstraints
from .supportFunc_ import supportFunc_
from .distance import distance # Import the new distance function
from .isFullDim import isFullDim
from .box import box

# Attach methods to the class
Polytope.zonotope = zonotope
Polytope.contains_ = contains_
Polytope.dim = dim
Polytope.center = center
Polytope.ellipsoid = ellipsoid
Polytope.isBounded = isBounded
Polytope.isemptyobject = isemptyobject
Polytope.is_empty = isemptyobject
Polytope.interval = interval
Polytope.empty = staticmethod(empty)
Polytope.Inf = staticmethod(Inf)
Polytope.origin = staticmethod(origin)
Polytope.copy = copy
Polytope.display = display
Polytope.mtimes = mtimes
Polytope.plus = plus
Polytope.project = project
Polytope.representsa_ = representsa_
Polytope.constraints = constraints
Polytope.vertices_ = vertices_
Polytope.generate_random = staticmethod(generate_random)
Polytope.get_print_set_info = get_print_set_info
Polytope.enclose_points = staticmethod(enclose_points)
Polytope.isIntersecting_ = isIntersecting_
Polytope.normalizeConstraints = normalizeConstraints
Polytope.supportFunc_ = supportFunc_
Polytope.distance = distance # Attach the distance method
Polytope.isFullDim = isFullDim
Polytope.box = box

# Attach operator overloads
Polytope.__contains__ = lambda self, other: contains_(self, other)
Polytope.__rcontains__ = lambda self, other: contains_(other, self)
Polytope.__str__ = display
Polytope.__add__ = lambda self, other: plus(self, other)
Polytope.__radd__ = lambda self, other: plus(other, self)
Polytope.__sub__ = lambda self, other: minus(self, other)
Polytope.__rsub__ = lambda self, other: rminus(self, other)
Polytope.__mul__ = lambda self, other: mtimes(self, other)
Polytope.__rmul__ = lambda self, other: mtimes(other, self)
Polytope.__matmul__ = lambda self, other: mtimes(self, other)
Polytope.__rmatmul__ = lambda self, other: mtimes(other, self)

__all__ = [
    'Polytope',
    'center',
    'contains_',
    'constraints',
    'copy',
    'dim',
    'display',
    'ellipsoid',
    'empty',
    'enclose_points',
    'generate_random',
    'get_print_set_info',
    'Inf',
    'isBounded',
    'isIntersecting_',
    'isemptyobject',
    'interval',
    'minus',
    'mtimes',
    'normalizeConstraints',
    'origin',
    'plus',
    'project',
    'representsa_',
    'rminus',
    'supportFunc_',
    'vertices_',
    'zonotope',
    'distance',
    'isFullDim',
    'box',
] 