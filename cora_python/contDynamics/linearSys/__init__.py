"""
linearSys package - Linear time-invariant systems

This package provides the LinearSys class for representing and working with
linear time-invariant dynamical systems.
"""

from .linearSys import LinearSys
from .display import display
from .eq import eq, isequal
from .ne import ne
from .generateRandom import generateRandom

# Attach methods to the LinearSys class
LinearSys.display = lambda self: display(self)
LinearSys.eq = lambda self, other, tol=None: eq(self, other, tol)
LinearSys.isequal = lambda self, other, tol=None: isequal(self, other, tol)
LinearSys.ne = lambda self, other, tol=None: ne(self, other, tol)

__all__ = ['LinearSys', 'display', 'eq', 'isequal', 'ne', 'generateRandom'] 