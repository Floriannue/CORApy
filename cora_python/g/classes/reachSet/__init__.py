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
from .isemptyobject import isemptyobject
from .plus import plus
from .minus import minus
from .times import times
from .mtimes import mtimes
from .eq import eq
from .ne import ne
from .uminus import uminus
from .uplus import uplus
from .plot import plot
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
    'isemptyobject',
    'plus',
    'minus',
    'times',
    'mtimes',
    'eq',
    'ne',
    'uminus',
    'uplus',
    'plot',
    'initReachSet'
] 