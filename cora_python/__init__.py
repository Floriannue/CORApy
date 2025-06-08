"""
CORA Python - Python translation of the CORA toolbox

This package provides Python implementations of the CORA (COntinuous Reachability Analysis)
toolbox functionality, including continuous set representations and operations.
"""

from . import contSet
from . import contDynamics
from .contSet import interval
from .contDynamics import LinearSys
from .g.functions.matlab.validate.postprocessing.CORAerror import CORAError

__version__ = "2025.1.0"
__all__ = ['contSet', 'contDynamics', 'interval', 'LinearSys', 'CORAError'] 