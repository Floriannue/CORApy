"""
Zonotope package - exports zonotope class and all its methods

This package contains the zonotope class implementation and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .zonotope import Zonotope
from .abs_op import abs_op
from .and_ import and_
from .box import box
from .plus import plus
from .minus import minus
from .times import times
from .uminus import uminus
from .isequal import isequal
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
from .isBounded import isBounded
from .copy import copy
from .convHull_ import convHull_
from .enclose import enclose
from .reduce import reduce
from .minnorm import minnorm
from .enclosePoints import enclosePoints
from .boundaryPoint import boundaryPoint
from .supportFunc_ import supportFunc_
from .radius import radius
from .rank import rank
from .volume_ import volume_
from .capsule import capsule
from .cartProd_ import cartProd_
from .conZonotope import conZonotope
from .polytope import polytope
from .polyZonotope import polyZonotope
from .zonoBundle import zonoBundle
from .generators import generators
from .isIntersecting_ import isIntersecting_
from .isFullDim import isFullDim
from .generatorLength import generatorLength
from .getPrintSetInfo import getPrintSetInfo
from .ellipsoid import ellipsoid
from .quadMap import quadMap
from .constrSat import constrSat

# Attach methods to the class
Zonotope.abs = abs_op
Zonotope.and_ = and_
Zonotope.__abs__ = abs_op
Zonotope.__and__ = and_
Zonotope.box = box
Zonotope.plus = plus
Zonotope.__add__ = plus
Zonotope.__radd__ = plus
Zonotope.minus = minus
Zonotope.__sub__ = minus
Zonotope.__mul__ = times
Zonotope.__rmul__ = times
Zonotope.uminus = uminus
Zonotope.__neg__ = uminus
Zonotope.__eq__ = isequal
Zonotope.__matmul__ = mtimes
Zonotope.__rmatmul__ = lambda self, other: mtimes(other, self)
Zonotope.dim = dim
Zonotope.empty = empty
Zonotope.origin = origin
Zonotope.isemptyobject = isemptyobject
Zonotope.is_empty = isemptyobject
Zonotope.display = display
Zonotope.randPoint_ = randPoint_
Zonotope.vertices_ = vertices_
Zonotope.project = project
Zonotope.mtimes = mtimes
Zonotope.center = center
Zonotope.representsa_ = representsa_
Zonotope.compact_ = compact_
Zonotope.interval = interval
Zonotope.contains_ = contains_
Zonotope.norm_ = norm_
Zonotope.zonotopeNorm = zonotopeNorm
Zonotope.isBounded = isBounded
Zonotope.copy = copy
Zonotope.convHull_ = convHull_
Zonotope.enclose = enclose
Zonotope.reduce = reduce
Zonotope.minnorm = minnorm
Zonotope.boundaryPoint = boundaryPoint
Zonotope.supportFunc_ = supportFunc_
Zonotope.radius = radius
Zonotope.rank = rank
Zonotope.volume_ = volume_
Zonotope.capsule = capsule
Zonotope.cartProd_ = cartProd_
Zonotope.conZonotope = conZonotope
Zonotope.polytope = polytope
Zonotope.polyZonotope = polyZonotope
Zonotope.zonoBundle = zonoBundle
Zonotope.generators = generators
Zonotope.isIntersecting_ = isIntersecting_
Zonotope.isFullDim = isFullDim
Zonotope.generatorLength = generatorLength
Zonotope.getPrintSetInfo = getPrintSetInfo
Zonotope.ellipsoid = ellipsoid
Zonotope.quadMap = quadMap
Zonotope.constrSat = constrSat

# Attach static methods
Zonotope.enclosePoints = staticmethod(enclosePoints)

__all__ = ['Zonotope', 'abs_op', 'and_', 'box', 'plus', 'minus', 'times', 'uminus', 'isequal', 'mtimes', 'dim', 'empty', 'origin', 'isemptyobject', 'display', 'randPoint_', 'vertices_', 'project', 'center', 'representsa_', 'compact_', 'interval', 'contains_', 'norm_', 'zonotopeNorm', 'isBounded', 'copy', 'convHull_', 'enclose', 'reduce', 'minnorm', 'enclosePoints', 'boundaryPoint', 'supportFunc_', 'radius', 'rank', 'volume_', 'capsule', 'cartProd_', 'conZonotope', 'polytope', 'polyZonotope', 'zonoBundle', 'generators', 'isIntersecting_', 'isFullDim', 'generatorLength', 'getPrintSetInfo', 'ellipsoid', 'quadMap', 'constrSat'] 