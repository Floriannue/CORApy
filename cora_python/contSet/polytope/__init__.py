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
from .display import display, display_
from .mtimes import mtimes
from .plus import plus
from .project import project
from .representsa_ import representsa_
from .constraints import constraints
from .vertices_ import vertices_
from .polyZonotope import polyZonotope
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
from .spectraShadow import spectraShadow
from .convHull_ import convHull_
from .compact_ import compact_
from .isequal import isequal
from .le import le
from .or_ import or_
from .not_ import not_
from .and_ import and_
from .cartProd_ import cartProd_
from .volume_ import volume_
from .reduceOverDomain import reduceOverDomain
from .projectHighDim import projectHighDim
from .randPoint_ import randPoint_
from .lift_ import lift_
from .hausdorffDist import hausdorffDist
from .zonoBundle import zonoBundle
from .conZonotope import conZonotope
from .eventFcn import eventFcn
from .enclose import enclose
from .mldivide import mldivide
from .setVertices import setVertices
from .conPolyZono import conPolyZono
from .levelSet import levelSet
from .matPolytope import matPolytope
from .minkDiff import minkDiff

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
Polytope.display_ = display_
Polytope.mtimes = mtimes
Polytope.plus = plus
Polytope.project = project
Polytope.representsa_ = representsa_
Polytope.constraints = constraints
Polytope.vertices_ = vertices_
Polytope.polyZonotope = polyZonotope
Polytope.generate_random = staticmethod(generate_random)
Polytope.get_print_set_info = get_print_set_info
Polytope.enclose_points = staticmethod(enclose_points)
Polytope.isIntersecting_ = isIntersecting_
Polytope.normalizeConstraints = normalizeConstraints
Polytope.supportFunc_ = supportFunc_
Polytope.distance = distance # Attach the distance method
Polytope.isFullDim = isFullDim
Polytope.box = box
Polytope.conPolyZono = conPolyZono
Polytope.spectraShadow = spectraShadow
Polytope.and_ = and_
Polytope.cartProd_ = cartProd_
Polytope.volume_ = volume_
Polytope.convHull_ = convHull_
Polytope.compact_ = compact_
Polytope.isequal = isequal
Polytope.le = le
Polytope.or_ = or_
Polytope.not_ = not_
Polytope.reduceOverDomain = reduceOverDomain
Polytope.projectHighDim = projectHighDim
Polytope.randPoint_ = randPoint_
Polytope.lift_ = lift_
Polytope.hausdorffDist = hausdorffDist
Polytope.zonoBundle = zonoBundle
Polytope.conZonotope = conZonotope
Polytope.eventFcn = eventFcn
Polytope.enclose = enclose
Polytope.mldivide = mldivide
Polytope.__truediv__ = None
Polytope.__rtruediv__ = None
Polytope.__floordiv__ = mldivide
Polytope.__rfloordiv__ = mldivide
Polytope.__mod__ = None
Polytope.__rmod__ = None
Polytope.__rshift__ = None
Polytope.__lshift__ = None
Polytope.__matmul__ = None
Polytope.__rmatmul__ = None
Polytope.__and__ = Polytope.and_
Polytope.__or__ = Polytope.or_
Polytope.__sub__ = None
Polytope.__rsub__ = None
Polytope.__truediv__ = None
Polytope.__rtruediv__ = None
Polytope.setVertices = setVertices
Polytope.levelSet = levelSet
Polytope.matPolytope = matPolytope # Attach the matPolytope method
Polytope.minkDiff = minkDiff # Attach the minkDiff method

# Attach operator overloads
Polytope.__contains__ = lambda self, other: contains_(self, other)
Polytope.__rcontains__ = lambda self, other: contains_(other, self)
Polytope.__str__ = lambda self: display_(self)
Polytope.__add__ = lambda self, other: plus(self, other)
Polytope.__radd__ = lambda self, other: plus(other, self)
Polytope.__sub__ = lambda self, other: minus(self, other)
Polytope.__rsub__ = lambda self, other: rminus(self, other)
Polytope.__mul__ = lambda self, other: mtimes(self, other)
Polytope.__rmul__ = lambda self, other: mtimes(other, self)
Polytope.__matmul__ = lambda self, other: mtimes(self, other)
Polytope.__rmatmul__ = lambda self, other: mtimes(other, self)
Polytope.__eq__ = isequal
Polytope.__ne__ = lambda self, other: not isequal(self, other)
Polytope.__le__ = lambda self, other: le(self, other)

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
	'spectraShadow',
	'and_',
	'cartProd_',
	'volume_',
	'convHull_',
	'compact_',
	'isequal',
	'reduceOverDomain',
	'projectHighDim',
	'randPoint_',
	'lift_',
	'hausdorffDist',
    'zonoBundle',
    'conZonotope',
    'eventFcn',
    'enclose',
    'mldivide',
    'setVertices',
    'conPolyZono',
    'levelSet',
    'matPolytope',
    'minkDiff',
] 