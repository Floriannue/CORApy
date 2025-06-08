"""
contDynamics package - Base class for continuous dynamics

This package provides the base ContDynamics class that serves as the foundation
for all continuous dynamical systems in CORA.
"""

from .contDynamics import ContDynamics
from .simulateRandom import simulateRandom

# Attach the simulateRandom method to the ContDynamics class
ContDynamics.simulateRandom = simulateRandom

__all__ = ['ContDynamics', 'simulateRandom'] 