"""
contSet package - Base class for all continuous sets

This package provides the contSet abstract base class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

# Import the main ContSet class
from .contSet import ContSet
try:
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
except ImportError:
    # Fallback for when running from within the cora_python directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Import all method implementations
from .plot import plot
from .plot1D import plot1D
from .plot2D import plot2D
from .plot3D import plot3D

# Import core infrastructure functions
from .representsa import representsa
from .representsa_ import representsa_
from .representsa_emptyObject import representsa_emptyObject
from .isemptyobject import isemptyobject
from .dim import dim
from .center import center
from .contains import contains
from .contains_ import contains_

# Import mathematical operations
from .generateRandom import generateRandom
from .times import times
from .decompose import decompose
from .project import project

# Import comparison operations
from .eq import eq
from .ne import ne
from .isequal import isequal
from .isempty import isempty

# Import display and utility functions
from .display import display
from .copy import copy

# Import static methods
from .empty import empty
from .Inf import Inf

# Import arithmetic operations
from .plus import plus
from .minus import minus
from .uminus import uminus
from .uplus import uplus
from .mtimes import mtimes

# Import property checking functions
from .isBounded import isBounded

# Import vertex computation
from .vertices import vertices
from .vertices_ import vertices_

# Import volume and norm computation
from .volume import volume
from .volume_ import volume_
from .norm import norm
from .norm_ import norm_

# Import set operations
from .and_op import and_op
from .and_ import and_
from .or_op import or_op
from .reorder import reorder

# Import convex hull operations
from .convHull import convHull
from .convHull_ import convHull_

# Import support function operations
from .supportFunc import supportFunc
from .supportFunc_ import supportFunc_

# Import random point generation
from .randPoint import randPoint
from .randPoint_ import randPoint_

# Import intersection checking
from .isIntersecting import isIntersecting
from .isIntersecting_ import isIntersecting_

# Import property checking functions
from .isFullDim import isFullDim
from .isZero import isZero

# Import geometric operations
from .enlarge import enlarge
from .lift import lift
from .lift_ import lift_
from .affine import affine
from .boundaryPoint import boundaryPoint
from .cartProd import cartProd
from .cartProd_ import cartProd_
from .enclose import enclose
from .emptySet import emptySet
from .fullspace import fullspace
from .getPrintSetInfo import getPrintSetInfo
from .plotRandPoint import plotRandPoint
from .zonotope import zonotope
from .capsule import capsule
from .conPolyZono import conPolyZono
from .conZonotope import conZonotope
from .ellipsoid import ellipsoid
from .interval import interval
from .levelSet import levelSet
from .polygon import polygon
from .polytope import polytope
from .polyZonotope import polyZonotope
from .probZonotope import probZonotope
from .zonoBundle import zonoBundle
from .isZonotope import isZonotope
from .isPolytope import isPolytope
from .isInterval import isInterval
from .isParallelotope import isParallelotope
from .isHyperplane import isHyperplane
from .isConHyperplane import isConHyperplane
from .printSet import printSet
from .projVertices import projVertices
from .projectHighDim import projectHighDim
from .projectHighDim_ import projectHighDim_
from .projectOnHyperplane import projectOnHyperplane
from .recompose import recompose

# Import compactification operations
from .compact import compact
from .compact_ import compact_

# Import additional operations
from .reduce import reduce
from .origin import origin
from .minkDiff import minkDiff
from .linComb import linComb
from .quadMap import quadMap
from .cubMap import cubMap

#import static methods
from .enclosePoints import enclosePoints
from .generateRandom import generateRandom
from .initEmptySet import initEmptySet
from .empty import empty
from .Inf import Inf

# Attach methods to the ContSet class
# This makes methods available as obj.method() calls

# Core methods
ContSet.plot = plot
ContSet.center = center
ContSet.contains = contains
ContSet.contains_ = contains_
ContSet.copy = copy
ContSet.display = display

# Mathematical operations
ContSet.times = times
ContSet.decompose = decompose
ContSet.project = project

# Arithmetic operations - attached as operators
ContSet.__add__ = lambda self, other: plus(self, other)
ContSet.__radd__ = lambda self, other: plus(other, self)
ContSet.__sub__ = lambda self, other: minus(self, other)
ContSet.__rsub__ = lambda self, other: minus(other, self)
ContSet.__neg__ = lambda self: uminus(self)
ContSet.__pos__ = lambda self: uplus(self)
ContSet.__mul__ = lambda self, other: mtimes(other, self)  # Note: MATLAB style
ContSet.__rmul__ = lambda self, other: mtimes(other, self)

# Comparison operations
ContSet.__eq__ = lambda self, other: eq(self, other)
ContSet.__ne__ = lambda self, other: ne(self, other)
ContSet.isequal = isequal
ContSet.isempty = isempty

# Type checking and properties
ContSet.representsa = representsa
ContSet.representsa_ = representsa_
ContSet.representsa_emptyObject = representsa_emptyObject
ContSet.isBounded = isBounded
ContSet.isFullDim = isFullDim
ContSet.isZero = isZero

# Geometric operations
ContSet.vertices = vertices
ContSet.vertices_ = vertices_
ContSet.volume = volume
ContSet.volume_ = volume_
ContSet.norm = norm
ContSet.norm_ = norm_

# Set operations
ContSet.__and__ = lambda self, other: and_op(self, other)
ContSet.__or__ = lambda self, other: or_op(self, other)
ContSet.and_op = and_op  # Public 'and' function (can't use 'and' as it's Python keyword)
ContSet.and_ = and_
ContSet.or_op = or_op

# Convex hull operations
ContSet.convHull = convHull
ContSet.convHull_ = convHull_

# Support function operations
ContSet.supportFunc = supportFunc
ContSet.supportFunc_ = supportFunc_

# Random operations
ContSet.randPoint = randPoint
ContSet.randPoint_ = randPoint_

# Intersection checking
ContSet.isIntersecting = isIntersecting
ContSet.isIntersecting_ = isIntersecting_

# Geometric transformations
ContSet.enlarge = enlarge
ContSet.lift = lift
ContSet.lift_ = lift_
ContSet.affine = affine
ContSet.boundaryPoint = boundaryPoint
ContSet.cartProd = cartProd
ContSet.cartProd_ = cartProd_
ContSet.enclose = enclose
ContSet.emptySet = emptySet
ContSet.fullspace = fullspace
ContSet.getPrintSetInfo = getPrintSetInfo
ContSet.plotRandPoint = plotRandPoint
ContSet.zonotope = zonotope
ContSet.capsule = capsule
ContSet.conPolyZono = conPolyZono
ContSet.conZonotope = conZonotope
ContSet.ellipsoid = ellipsoid
ContSet.interval = interval
ContSet.levelSet = levelSet
ContSet.polygon = polygon
ContSet.polytope = polytope
ContSet.polyZonotope = polyZonotope
ContSet.probZonotope = probZonotope
ContSet.zonoBundle = zonoBundle
ContSet.isZonotope = isZonotope
ContSet.isPolytope = isPolytope
ContSet.isInterval = isInterval
ContSet.isParallelotope = isParallelotope
ContSet.isHyperplane = isHyperplane
ContSet.isConHyperplane = isConHyperplane
ContSet.printSet = printSet
ContSet.projVertices = projVertices
ContSet.projectHighDim = projectHighDim
ContSet.projectHighDim_ = projectHighDim_
ContSet.projectOnHyperplane = projectOnHyperplane
ContSet.recompose = recompose
ContSet.compact = compact
ContSet.compact_ = compact_
ContSet.reduce = reduce

# Additional operations
ContSet.origin = origin
ContSet.minkDiff = minkDiff
ContSet.linComb = linComb
ContSet.quadMap = quadMap
ContSet.cubMap = cubMap
ContSet.reorder = reorder

#static methods
ContSet.enclosePoints = staticmethod(enclosePoints)
ContSet.generateRandom = staticmethod(generateRandom)
ContSet.initEmptySet = staticmethod(initEmptySet)
ContSet.empty = staticmethod(empty)
ContSet.Inf = staticmethod(Inf)

# Export the ContSet class and all methods
__all__ = [
    'ContSet',
    'CORAerror',
    'plot',
    'plot1D',
    'plot2D',
    'plot3D',
    'representsa',
    'representsa_',
    'representsa_emptyObject',
    'isemptyobject', 
    'dim',
    'center',
    'contains',
    'contains_',
    'generateRandom',
    'times',
    'decompose',
    'project',
    'eq',
    'ne',
    'isequal',
    'isempty',
    'display',
    'copy',
    'empty',
    'Inf',
    'plus',
    'minus',
    'uminus',
    'uplus',
    'mtimes',
    'isBounded',
    'vertices',
    'vertices_',
    'volume',
    'volume_',
    'norm',
    'norm_',
    'and_op',
    'and_',
    'or_op',
    'reorder',
    'convHull',
    'convHull_',
    'supportFunc',
    'supportFunc_',
    'randPoint',
    'randPoint_',
    'isIntersecting',
    'isIntersecting_',
    'isFullDim',
    'isZero',
    'enlarge',
    'lift',
    'lift_',
    'affine',
    'boundaryPoint',
    'cartProd',
    'cartProd_',
    'enclose',
    'emptySet',
    'fullspace',
    'getPrintSetInfo',
    'plotRandPoint',
    'zonotope',
    'capsule',
    'conPolyZono',
    'conZonotope',
    'ellipsoid',
    'interval',
    'levelSet',
    'polygon',
    'polytope',
    'polyZonotope',
    'probZonotope',
    'zonoBundle',
    'isZonotope',
    'isPolytope',
    'isInterval',
    'isParallelotope',
    'isHyperplane',
    'isConHyperplane',
    'printSet',
    'projVertices',
    'projectHighDim',
    'projectHighDim_',
    'projectOnHyperplane',
    'recompose',
    'compact',
    'compact_',
    'reduce',
    'origin',
    'minkDiff',
    'linComb',
    'quadMap',
    'cubMap'
] 