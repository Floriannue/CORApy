"""
reachSet package - Class for storing reachable sets

This package provides the reachSet class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
"""

# Import the main ReachSet class
from .reachSet import ReachSet

# Import all method implementations
from .query import query
from .find import find
from .project import project
from .add import add
from .append import append
from .children import children
from .contains import contains
from .isequal import isequal
from .isemptyobject import isemptyobject
from .order import order
from .plus import plus
from .minus import minus
from .shiftTime import shiftTime
from .times import times
from .mtimes import mtimes
from .eq import eq
from .ne import ne
from .uminus import uminus
from .uplus import uplus
from .plot import plot
from .plotOverTime import plotOverTime
from .plotTimeStep import plotTimeStep
from .plotAsGraph import plotAsGraph
from .modelChecking import modelChecking
from .initReachSet import initReachSet

# Attach methods to the ReachSet class
ReachSet.query = query
ReachSet.find = find
ReachSet.project = project
ReachSet.add = add
ReachSet.append = append
ReachSet.children = children
ReachSet.contains = contains
ReachSet.isequal = isequal
ReachSet.isemptyobject = isemptyobject
ReachSet.order = order
ReachSet.shiftTime = shiftTime
ReachSet.plot = plot
ReachSet.plotOverTime = plotOverTime
ReachSet.plotTimeStep = plotTimeStep
ReachSet.plotAsGraph = plotAsGraph
ReachSet.modelChecking = modelChecking

# Attach arithmetic methods
ReachSet.plus = plus
ReachSet.minus = minus
ReachSet.times = times
ReachSet.mtimes = mtimes
ReachSet.uminus = uminus
ReachSet.uplus = uplus
ReachSet.eq = eq
ReachSet.ne = ne

# Attach operator overloads
ReachSet.__add__ = lambda self, other: plus(self, other)
ReachSet.__radd__ = lambda self, other: plus(other, self)
ReachSet.__sub__ = lambda self, other: minus(self, other)
ReachSet.__rsub__ = lambda self, other: minus(other, self)
ReachSet.__mul__ = lambda self, other: times(self, other)
ReachSet.__rmul__ = lambda self, other: times(other, self)
ReachSet.__matmul__ = lambda self, other: mtimes(self, other)
ReachSet.__rmatmul__ = lambda self, other: mtimes(other, self)
ReachSet.__eq__ = lambda self, other: eq(self, other)
ReachSet.__ne__ = lambda self, other: ne(self, other)
ReachSet.__neg__ = lambda self: uminus(self)
ReachSet.__pos__ = lambda self: uplus(self)

# Attach static methods
ReachSet.initReachSet = staticmethod(initReachSet)

# Export the ReachSet class and all methods
__all__ = [
    'ReachSet',
    'query',
    'find', 
    'project',
    'add',
    'append',
    'children',
    'contains',
    'isequal',
    'isemptyobject',
    'order',
    'plus',
    'minus',
    'shiftTime',
    'times',
    'mtimes',
    'eq',
    'ne',
    'uminus',
    'uplus',
    'plot',
    'plotOverTime',
    'plotTimeStep',
    'plotAsGraph',
    'modelChecking',
    'initReachSet'
] 