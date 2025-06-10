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
    'project'
] 