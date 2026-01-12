"""
transition - Transition package for hybrid automata

This package contains the Transition class and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .transition import Transition
from .isequal import isequal
from .isemptyobject import isemptyobject
from .display import display, display_
from .synchronize import synchronize
from .guard2polytope import guard2polytope
from .convGuard import convGuard
from .derivatives import derivatives
from .eventFcn import eventFcn

# Attach methods to the Transition class
Transition.isequal = isequal
Transition.isemptyobject = isemptyobject
Transition.display = display
Transition.display_ = display_
Transition.synchronize = synchronize
Transition.guard2polytope = guard2polytope
Transition.convGuard = convGuard
Transition.derivatives = derivatives
Transition.eventFcn = eventFcn

# Attach display_ to __str__
Transition.__str__ = lambda self: display_(self)

__all__ = ['Transition', 'isequal', 'isemptyobject', 'display', 'display_', 
           'synchronize', 'guard2polytope', 'convGuard', 'derivatives', 'eventFcn']


