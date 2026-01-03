"""
linearReset - LinearReset package for hybrid automata

This package contains the LinearReset class and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .linearReset import LinearReset
from .evaluate import evaluate
from .isemptyobject import isemptyobject
from .isequal import isequal

# Attach methods to the LinearReset class
LinearReset.evaluate = evaluate
LinearReset.isemptyobject = isemptyobject
LinearReset.isequal = isequal

__all__ = ['LinearReset', 'evaluate', 'isemptyobject', 'isequal']

