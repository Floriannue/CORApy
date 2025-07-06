"""
CORA Python - Python translation of the CORA toolbox

This package provides Python implementations of the CORA (COntinuous Reachability Analysis)
toolbox functionality, including continuous set representations and operations.
"""

__version__ = "2025.1.0"

# Import main modules
from . import contDynamics
from . import contSet
from . import g
from . import matrixSet
from . import specification

__all__ = ['contDynamics', 'contSet', 'g', 'matrixSet', 'specification'] 