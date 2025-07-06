"""
MatrixSet package - Abstract superclass for matrix set representations

This package provides the MatrixSet class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

# Import the main MatrixSet class
from .matrixSet import MatrixSet

# Import method implementations
from .getPrintSetInfo import getPrintSetInfo

# Attach methods to the MatrixSet class
MatrixSet.getPrintSetInfo = getPrintSetInfo

# Export the MatrixSet class and all methods
__all__ = [
    'MatrixSet',
    'getPrintSetInfo',
] 