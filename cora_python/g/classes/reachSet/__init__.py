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