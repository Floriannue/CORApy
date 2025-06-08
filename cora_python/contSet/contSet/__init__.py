"""
contSet package - Base class for all continuous sets

This package provides the contSet abstract base class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

# Import the main contSet class
from .contSet import contSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError

# Import all method implementations
from .plot import plot
from .plot1D import plot1D
from .plot2D import plot2D
from .plot3D import plot3D

# Export the contSet class and all methods
__all__ = [
    'contSet',
    'CORAError',
    'plot',
    'plot1D',
    'plot2D',
    'plot3D'
] 