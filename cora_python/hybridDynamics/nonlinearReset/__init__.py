"""
nonlinearReset - NonlinearReset package for hybrid automata

This package contains the NonlinearReset class and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .nonlinearReset import NonlinearReset
from .isequal import isequal
from .derivatives import derivatives

# Attach methods to the NonlinearReset class
NonlinearReset.isequal = isequal
NonlinearReset.derivatives = derivatives

__all__ = ['NonlinearReset', 'isequal', 'derivatives']

