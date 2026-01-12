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
from .display import display, display_
from .derivatives import derivatives
from .priv_isFinalLocation import priv_isFinalLocation
from .private.priv_flowDerivatives import priv_flowDerivatives

# Attach methods to the HybridAutomaton class
HybridAutomaton.reach = reach
HybridAutomaton.isequal = isequal
HybridAutomaton.isemptyobject = isemptyobject
HybridAutomaton.display = display
HybridAutomaton.display_ = display_
HybridAutomaton.derivatives = derivatives
HybridAutomaton.priv_isFinalLocation = priv_isFinalLocation

# Attach display_ to __str__
HybridAutomaton.__str__ = lambda self: display_(self)

# Private methods (not attached as instance methods, but available for internal use)
# priv_flowDerivatives is used internally by reach

__all__ = ['HybridAutomaton', 'reach', 'isequal', 'isemptyobject', 'display', 'display_', 
           'derivatives', 'priv_isFinalLocation', 'priv_flowDerivatives']

