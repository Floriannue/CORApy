"""
contSet package - Base class for all continuous sets

This package provides the contSet abstract base class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

# Import the main ContSet class
from .contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError

# Import all method implementations
from .plot import plot
from .plot1D import plot1D
from .plot2D import plot2D
from .plot3D import plot3D

# Import core infrastructure functions
from .representsa import representsa
from .representsa_ import representsa_
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

# Export the ContSet class and all methods
__all__ = [
    'ContSet',
    'CORAError',
    'plot',
    'plot1D',
    'plot2D',
    'plot3D',
    'representsa',
    'representsa_',
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
    'compact',
    'compact_',
    'reduce',
    'origin',
    'minkDiff',
    'linComb',
    'quadMap',
    'cubMap'
] 