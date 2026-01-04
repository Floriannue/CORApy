"""
hybridAutomaton - Hybrid automaton package for CORA Python

This package contains the HybridAutomaton class and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

# Import the main HybridAutomaton class
from .hybridAutomaton import HybridAutomaton

# Import method implementations
from .reach import reach
from .isequal import isequal
from .isemptyobject import isemptyobject
from .display import display
from .private.priv_flowDerivatives import priv_flowDerivatives

# Attach methods to the HybridAutomaton class
HybridAutomaton.reach = reach
HybridAutomaton.isequal = isequal
HybridAutomaton.isemptyobject = isemptyobject
HybridAutomaton.display = display

# Private methods (not attached as instance methods, but available for internal use)
# priv_flowDerivatives is used internally by reach

__all__ = ['HybridAutomaton', 'reach', 'isequal', 'isemptyobject', 'display', 'priv_flowDerivatives']

