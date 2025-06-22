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

# Attach all methods to the Interval class
Interval.plus = plus
Interval.minus = minus
Interval.mtimes = mtimes
Interval.times = times
# dim and is_empty are implemented directly in class (required by ContSet)
Interval.isemptyobject = isemptyobject
Interval.representsa_ = representsa_
Interval.isequal = isequal
Interval.__eq__ = isequal  # Attach to == operator
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

# Attach static methods
Interval.empty = staticmethod(empty)
Interval.Inf = staticmethod(Inf)
Interval.origin = staticmethod(origin)
Interval.generateRandom = staticmethod(generateRandom)

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
] 
