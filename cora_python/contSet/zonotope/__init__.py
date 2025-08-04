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
from .reduceUnderApprox import reduceUnderApprox
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
from .generateRandom import generateRandom
from .intersectStrip import intersectStrip
from .lift_ import lift_
from .minkDiff import minkDiff
from .or_ import or_
from .underapproximate import underapproximate
from .dH2box import dH2box
from .dominantDirections import dominantDirections
from .spectraShadow import spectraShadow
from .taylm import taylm
from .volumeRatio import volumeRatio
from .filterOut import filterOut

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
Zonotope.__mul__ = mtimes
Zonotope.__rmul__ = lambda self, other: mtimes(other, self)
Zonotope.uminus = uminus
Zonotope.__neg__ = uminus
Zonotope.__eq__ = isequal
Zonotope.__matmul__ = mtimes
Zonotope.__rmatmul__ = lambda self, other: mtimes(other, self)
Zonotope.dim = dim
Zonotope.empty = empty
Zonotope.origin = staticmethod(origin)
Zonotope.isemptyobject = isemptyobject
Zonotope.is_empty = isemptyobject
Zonotope.display = display
Zonotope.randPoint_ = randPoint_
Zonotope.vertices_ = vertices_
Zonotope.project = project
Zonotope.mtimes = mtimes
# Zonotope.center = center
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
Zonotope.reduceUnderApprox = reduceUnderApprox
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
Zonotope.intersectStrip = intersectStrip
Zonotope.lift_ = lift_
Zonotope.minkDiff = minkDiff
Zonotope.or_ = or_
Zonotope.__or__ = or_
Zonotope.underapproximate = underapproximate
Zonotope.dH2box = dH2box
Zonotope.dominantDirections = dominantDirections
Zonotope.spectraShadow = spectraShadow
Zonotope.taylm = taylm
Zonotope.volumeRatio = volumeRatio
Zonotope.filterOut = filterOut

# Attach static methods
Zonotope.enclosePoints = staticmethod(enclosePoints)
Zonotope.generateRandom = staticmethod(generateRandom)

__all__ = ['Zonotope', 'abs_op', 'and_', 'box', 'plus', 'minus', 'uminus', 'isequal', 'mtimes', 'dim', 'empty', 'origin', 'isemptyobject', 'display', 'randPoint_', 'vertices_', 'project', 'center', 'representsa_', 'compact_', 'interval', 'contains_', 'norm_', 'zonotopeNorm', 'isBounded', 'copy', 'convHull_', 'enclose', 'reduce', 'reduceUnderApprox', 'minnorm', 'enclosePoints', 'boundaryPoint', 'supportFunc_', 'radius', 'rank', 'volume_', 'capsule', 'cartProd_', 'conZonotope', 'polytope', 'polyZonotope', 'zonoBundle', 'generators', 'isIntersecting_', 'isFullDim', 'generatorLength', 'getPrintSetInfo', 'ellipsoid', 'quadMap', 'constrSat', 'generateRandom', 'intersectStrip', 'lift_', 'minkDiff', 'or_', 'underapproximate', 'dH2box', 'dominantDirections', 'spectraShadow', 'taylm', 'volumeRatio', 'filterOut'] 