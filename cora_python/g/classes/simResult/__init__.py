"""
simResult package - Class that stores simulation results

This package provides the simResult class and all its methods.
Each method is implemented in a separate file following the MATLAB structure.

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
"""

# Import the main SimResult class
from .simResult import SimResult

# Import all method implementations
from .find import find
from .add import add
from .extractHits import extractHits
from .isemptyobject import isemptyobject
from .plus import plus
from .minus import minus
from .mtimes import mtimes
from .times import times
from .uminus import uminus
from .uplus import uplus
from .plot import plot
from .plotOverTime import plotOverTime
from .plotTimeStep import plotTimeStep
from .printSimResult import printSimResult
from .monitorSTL import monitorSTL

# Attach methods to the SimResult class
SimResult.find = find
SimResult.add = add
SimResult.extractHits = extractHits
SimResult.is_empty = isemptyobject
SimResult.isemptyobject = isemptyobject
SimResult.plot = plot
SimResult.plotOverTime = plotOverTime
SimResult.plotTimeStep = plotTimeStep
SimResult.printSimResult = printSimResult
SimResult.monitorSTL = monitorSTL

# Attach arithmetic methods
SimResult.plus = plus
SimResult.minus = minus
SimResult.times = times
SimResult.mtimes = mtimes
SimResult.uminus = uminus
SimResult.uplus = uplus

# Attach operator overloads
SimResult.__add__ = lambda self, other: plus(self, other)
SimResult.__radd__ = lambda self, other: plus(other, self)
SimResult.__sub__ = lambda self, other: minus(self, other)
SimResult.__rsub__ = lambda self, other: minus(other, self)
SimResult.__mul__ = lambda self, other: times(self, other)
SimResult.__rmul__ = lambda self, other: times(other, self)
SimResult.__matmul__ = lambda self, other: mtimes(self, other)
SimResult.__rmatmul__ = lambda self, other: mtimes(other, self)
SimResult.__neg__ = lambda self: uminus(self)
SimResult.__pos__ = lambda self: uplus(self)

# Export the SimResult class and all methods
__all__ = [
    'SimResult',
    'find',
    'add',
    'extractHits',
    'isemptyobject',
    'plus',
    'minus',
    'mtimes',
    'times',
    'uminus',
    'uplus',
    'plot',
    'plotOverTime',
    'plotTimeStep',
    'printSimResult',
    'monitorSTL'
] 