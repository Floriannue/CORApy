"""
linearParamSys package - Linear parametric systems

This package contains the LinearParamSys class and its methods.
"""

from .linearParamSys import LinearParamSys
from cora_python.contDynamics.linearSys.particularSolution_timeVarying import particularSolution_timeVarying
from cora_python.contDynamics.linearSys.particularSolution_constant import particularSolution_constant
from cora_python.contDynamics.linearSys.homogeneousSolution import homogeneousSolution

# Attach linearSys helper methods needed by linearParamSys workflows
LinearParamSys.particularSolution_timeVarying = particularSolution_timeVarying
LinearParamSys.particularSolution_constant = particularSolution_constant
LinearParamSys.homogeneousSolution = homogeneousSolution

__all__ = ['LinearParamSys']
