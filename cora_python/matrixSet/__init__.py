"""
MatrixSet package - Matrix set representations

This package provides matrix set representations for CORA.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

# Import matrix set classes
from .matrixSet import MatrixSet
from .intervalMatrix import IntervalMatrix

# Export all classes
__all__ = [
    'MatrixSet',
    'IntervalMatrix',
] 