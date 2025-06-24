"""
linearSys package - Linear time-invariant systems

This package provides the LinearSys class for representing and working with
linear time-invariant dynamical systems.
"""

from .linearSys import LinearSys
from .display import display
from .eq import eq
from .isequal import isequal
from .ne import ne
from .generateRandom import generateRandom
from .simulate import simulate
from .reach import reach
from .canonicalForm import canonicalForm
from .oneStep import oneStep
from cora_python.g.classes.taylorLinSys import TaylorLinSys
from .private.priv_reach_standard import priv_reach_standard
from .private.priv_reach_wrappingfree import priv_reach_wrappingfree
from .private.priv_outputSet_canonicalForm import priv_outputSet_canonicalForm
from .homogeneousSolution import homogeneousSolution
from .affineSolution import affineSolution
from ..contDynamics.simulateRandom import simulateRandom
from .particularSolution_constant import particularSolution_constant
from .particularSolution_timeVarying import particularSolution_timeVarying

# Attach methods to the LinearSys class
LinearSys.display = display
LinearSys.__eq__ = lambda self, other: eq(self, other) if isinstance(other, LinearSys) else False
LinearSys.__ne__ = lambda self, other: ne(self, other)
LinearSys.eq = eq
LinearSys.ne = ne
LinearSys.isequal = isequal
LinearSys.simulate = simulate
LinearSys.simulateRandom = simulateRandom
LinearSys.reach = reach
LinearSys.canonicalForm = canonicalForm
LinearSys.oneStep = oneStep
LinearSys.homogeneousSolution = homogeneousSolution
LinearSys.affineSolution = affineSolution
LinearSys.particularSolution_constant = particularSolution_constant
LinearSys.particularSolution_timeVarying = particularSolution_timeVarying

# Attach static methods
LinearSys.generateRandom = staticmethod(generateRandom)
LinearSys.taylorLinSys = lambda self, options=None: TaylorLinSys(self.A)

__all__ = ['LinearSys', 'display', 'eq', 'isequal', 'ne', 'generateRandom', 'simulate', 
           'simulateRandom', 'reach', 'canonicalForm', 'oneStep', 'TaylorLinSys', 'priv_reach_standard', 
           'priv_reach_wrappingfree', 'priv_outputSet_canonicalForm', 'homogeneousSolution', 'affineSolution'] 