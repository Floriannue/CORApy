"""
simResult package - Class that stores simulation results

This package provides the simResult class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
"""

# Import the main SimResult class
from .simResult import SimResult

# Import all method implementations
from .find import find
from .add import add
from .isemptyobject import isemptyobject
from .plus import plus
from .minus import minus
from .mtimes import mtimes

# Export the SimResult class and all methods
__all__ = [
    'SimResult',
    'find',
    'add',
    'isemptyobject',
    'plus',
    'minus',
    'mtimes'
] 