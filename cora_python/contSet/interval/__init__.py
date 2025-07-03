"""
Interval package - Real-valued intervals

This package provides the interval class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

# Import the main Interval class
from .interval import Interval

# Import all method implementations
from .plus import plus
from .minus import minus
from .mtimes import mtimes
from .times import times
from .dim import dim
from .isemptyobject import isemptyobject
from .representsa_ import representsa_
from .isequal import isequal
from .contains_ import contains_
from .center import center
from .rad import rad
from .empty import empty
from .Inf import Inf
from .origin import origin
from .project import project
from .is_bounded import is_bounded
from .vertices_ import vertices_
from .and_ import and_
from .randPoint_ import randPoint_
from .generateRandom import generateRandom
from .display import display
from .uminus import uminus
from .transpose import transpose
from .enclosePoints import enclosePoints
from .abs_op import abs_op

# Import comparison operators
from .le import le
from .lt import lt

# Import union operation
from .or_op import or_op

# Import concatenation operations  
from .horzcat import horzcat
from .vertcat import vertcat

# Import size operations
from .length import length
from .size import size

# Import mathematical operations
from .exp import exp
from .uplus import uplus
from .volume_ import volume_
from .sqrt import sqrt
from .log import log
from .cos import cos
from .sin import sin
from .min import min
from .max import max
from .infimum import infimum
from .supremum import supremum

# Import new mathematical operations
from .power import power
from .rdivide import rdivide
from .mpower import mpower
from .tan import tan
from .acos import acos
from .atan import atan
from .sinh import sinh
from .cosh import cosh
from .tanh import tanh
from .reshape import reshape
from .acosh import acosh
from .asin import asin
from .asinh import asinh
from .atanh import atanh
from .sign import sign
from .round_op import round_op
from .sum_op import sum_op
from .prod_op import prod_op
from .diag import diag
from .split import split
from .norm_ import norm_
from .tril import tril
from .triu import triu
from .copy import copy
from .isnan import isnan
from .isscalar import isscalar
from .issparse import issparse
from .kron import kron
from .lift_ import lift_
from .minkDiff import minkDiff

# Attach all methods to the Interval class
Interval.plus = plus
Interval.minus = minus
Interval.mtimes = mtimes
Interval.times = times
Interval.dim = dim
Interval.is_empty = isemptyobject
Interval.isemptyobject = isemptyobject
Interval.representsa_ = representsa_
Interval.isequal = isequal
Interval.contains_ = contains_
Interval.center = center
Interval.rad = rad
Interval.project = project
Interval.is_bounded = is_bounded
Interval.vertices_ = vertices_
Interval.and_ = and_
Interval.randPoint_ = randPoint_
Interval.display = display
Interval.uminus = uminus
Interval.transpose = transpose
Interval.le = le
Interval.lt = lt
Interval.or_op = or_op
Interval.horzcat = horzcat
Interval.vertcat = vertcat
Interval.length = length
Interval.size = size
Interval.abs = abs_op
Interval.exp = exp
Interval.uplus = uplus
Interval.volume_ = volume_
Interval.sqrt = sqrt
Interval.log = log
Interval.cos = cos
Interval.sin = sin
Interval.min = min
Interval.max = max
Interval.infimum = infimum
Interval.supremum = supremum
Interval.sign = sign

# Attach new mathematical operations
Interval.power = power
Interval.rdivide = rdivide
Interval.mpower = mpower
Interval.tan = tan
Interval.acos = acos
Interval.atan = atan
Interval.sinh = sinh
Interval.cosh = cosh
Interval.tanh = tanh
Interval.reshape = reshape
Interval.acosh = acosh
Interval.asin = asin
Interval.asinh = asinh
Interval.atanh = atanh
Interval.round = round_op
Interval.sum = sum_op
Interval.prod = prod_op
Interval.diag = diag
Interval.split = split
Interval.norm_ = norm_
Interval.tril = tril
Interval.triu = triu
Interval.copy = copy
Interval.isnan = isnan
Interval.isscalar = isscalar
Interval.issparse = issparse
Interval.kron = kron
Interval.lift_ = lift_
Interval.minkDiff = minkDiff

# Attach operator overloading
Interval.__eq__ = isequal  # == operator
Interval.__add__ = plus    # + operator
Interval.__radd__ = plus   # + operator (reverse)
Interval.__sub__ = minus   # - operator
Interval.__rsub__ = lambda self, other: minus(other, self)  # - operator (reverse)
Interval.__mul__ = times   # * operator (element-wise)
Interval.__rmul__ = times  # * operator (reverse)
Interval.__matmul__ = mtimes  # @ operator (matrix multiplication)
Interval.__rmatmul__ = lambda self, other: mtimes(other, self)  # @ operator (reverse)
Interval.__neg__ = uminus  # -obj (unary minus)
Interval.__le__ = le       # <= operator
Interval.__lt__ = lt       # < operator
Interval.__ge__ = lambda self, other: le(other, self)  # >= operator
Interval.__gt__ = lambda self, other: lt(other, self)  # > operator
Interval.__or__ = or_op    # | operator (union)
Interval.__ror__ = lambda self, other: or_op(other, self)  # | operator (reverse)
Interval.__and__ = and_    # & operator (intersection)
Interval.__rand__ = lambda self, other: and_(other, self)  # & operator (reverse)
Interval.__len__ = length
Interval.__abs__ = abs_op
Interval.__pos__ = uplus  # +obj (unary plus)
Interval.__pow__ = power  # ** operator (element-wise power)
Interval.__rpow__ = lambda self, other: power(other, self)  # ** operator (reverse)
Interval.__truediv__ = rdivide  # / operator (element-wise division)
Interval.__rtruediv__ = lambda self, other: rdivide(other, self)  # / operator (reverse)
Interval.__round__ = round_op
Interval.__sum__ = sum_op

# Attach static methods
Interval.empty = staticmethod(empty)
Interval.Inf = staticmethod(Inf)
Interval.origin = staticmethod(origin)
Interval.generateRandom = staticmethod(generateRandom)
Interval.enclosePoints = staticmethod(enclosePoints)

# Export the Interval class and all methods
__all__ = [
    'Interval',
    'plus',
    'minus',
    'mtimes',
    'times',
    'dim',
    'isemptyobject',
    'representsa_',
    'isequal',
    'contains_',
    'center',
    'rad',
    'empty',
    'Inf',
    'origin',
    'project',
    'is_bounded',
    'vertices_',
    'and_',
    'randPoint_',
    'generateRandom',
    'display',
    'uminus',
    'transpose',
    'enclosePoints',
    'abs_op',
    'le',
    'lt', 
    'or_op',
    'horzcat',
    'vertcat',
    'length',
    'size',
    'exp',
    'uplus',
    'volume_',
    'sqrt',
    'log',
    'cos',
    'sin',
    'min',
    'max',
    'infimum',
    'supremum',
    'sign',
    'round_op',
    'sum_op',
    'prod_op',
    'diag',
    'split',
    'norm_',
    'tril',
    'triu',
    'copy',
    'isnan',
    'isscalar',
    'issparse',
    'kron',
    'lift_',
    'minkDiff',
    'power',
    'rdivide', 
    'mpower',
    'tan',
    'acos',
    'atan',
    'sinh',
    'cosh',
    'tanh',
    'reshape',
    'acosh',
    'asin',
    'asinh',
    'atanh',
] 
