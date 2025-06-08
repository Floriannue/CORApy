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
from .simulate import simulate
from .reach import reach
from .canonicalForm import canonicalForm
from .oneStep import oneStep
from ...g.classes.taylorLinSys import TaylorLinSys
from .priv_reach_standard import priv_reach_standard
from .priv_reach_wrappingfree import priv_reach_wrappingfree
from .priv_outputSet_canonicalForm import priv_outputSet_canonicalForm

# Attach methods to the LinearSys class
LinearSys.display = lambda self: display(self)
LinearSys.eq = lambda self, other, tol=None: eq(self, other, tol)
LinearSys.isequal = lambda self, other, tol=None: isequal(self, other, tol)
LinearSys.ne = lambda self, other, tol=None: ne(self, other, tol)
LinearSys.simulate = lambda self, params, options=None: simulate(self, params, options)
LinearSys.reach = lambda self, params, options=None: reach(self, params, options)
LinearSys.canonicalForm = lambda self: canonicalForm(self)
LinearSys.oneStep = lambda self, params, options=None: oneStep(self, params, options)
LinearSys.taylorLinSys = lambda self, options=None: TaylorLinSys(self.A)

__all__ = ['LinearSys', 'display', 'eq', 'isequal', 'ne', 'generateRandom', 'simulate', 
           'reach', 'canonicalForm', 'oneStep', 'TaylorLinSys', 'priv_reach_standard', 
           'priv_reach_wrappingfree', 'priv_outputSet_canonicalForm'] 