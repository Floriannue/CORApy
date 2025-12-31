"""
nonlinearSys package - Nonlinear continuous-time systems

This package provides the NonlinearSys class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Matthias Althoff, Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
"""

from .initReach import initReach
from .linearize import linearize
from .post import post

__all__ = ['initReach', 'linearize', 'post']

