"""
contDynamics package - Base class for continuous dynamics

This package provides the base ContDynamics class that serves as the foundation
for all continuous dynamical systems in CORA.
"""

from .contDynamics import ContDynamics
from .simulateRandom import simulateRandom

# Import private functions
from .private.priv_simulateStandard import priv_simulateStandard
from .private.priv_simulateGaussian import priv_simulateGaussian
from .private.priv_simulateRRT import priv_simulateRRT
from .private.priv_simulateConstrainedRandom import priv_simulateConstrainedRandom

# Attach the simulateRandom method to the ContDynamics class
ContDynamics.simulateRandom = simulateRandom

__all__ = ['ContDynamics', 'simulateRandom', 'priv_simulateStandard', 
           'priv_simulateGaussian', 'priv_simulateRRT', 'priv_simulateConstrainedRandom'] 