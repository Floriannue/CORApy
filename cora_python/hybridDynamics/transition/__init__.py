"""
transition - Transition package for hybrid automata

This package contains the Transition class and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .transition import Transition
from .isequal import isequal
from .isemptyobject import isemptyobject
from .display import display, display_

# Attach methods to the Transition class
Transition.isequal = isequal
Transition.isemptyobject = isemptyobject
Transition.display = display
Transition.display_ = display_

# Attach display_ to __str__
Transition.__str__ = lambda self: display_(self)

__all__ = ['Transition', 'isequal', 'isemptyobject', 'display', 'display_']


