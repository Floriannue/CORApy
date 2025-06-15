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
from .isemptyobject import isemptyobject
from .plus import plus
from .minus import minus

# Export the ReachSet class and all methods
__all__ = [
    'ReachSet',
    'query',
    'find', 
    'project',
    'add',
    'isemptyobject',
    'plus',
    'minus'
] 