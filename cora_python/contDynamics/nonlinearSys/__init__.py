"""
nonlinearSys package - Nonlinear continuous-time systems

This package provides the NonlinearSys class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Matthias Althoff, Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
"""

from .initReach import initReach
from .initReach_adaptive import initReach_adaptive
from .linearize import linearize
from .post import post
from .nonlinearSys import NonlinearSys
from .display import display, display_

# Attach methods to the NonlinearSys class
NonlinearSys.initReach = initReach
NonlinearSys.initReach_adaptive = initReach_adaptive
NonlinearSys.linearize = linearize
NonlinearSys.post = post
NonlinearSys.display = display
NonlinearSys.display_ = display_

# Attach display_ to __str__
NonlinearSys.__str__ = lambda self: display_(self)

__all__ = ['NonlinearSys', 'initReach', 'initReach_adaptive', 'linearize', 'post', 'display', 'display_']

