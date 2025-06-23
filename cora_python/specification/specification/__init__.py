"""
Specification module for CORA

This module contains classes for temporal logic specifications
and their verification in reachability analysis.
"""

from .specification import Specification, create_safety_specification, create_invariant_specification, create_unsafe_specification
from .add import add
from .check import check
from .eq import eq
from .isequal import isequal
from .inverse import inverse
from .isempty import isempty
from .ne import ne
from .project import project
from .splitLogic import splitLogic
from .plot import plot
from .plotOverTime import plotOverTime
from .robustness import robustness
from .printSpec import printSpec

# Attach methods to the Specification class
Specification.__eq__ = lambda self, other: eq(self, other)
Specification.__ne__ = lambda self, other: ne(self, other)
Specification.__add__ = lambda self, other: add(self, other)
Specification.__radd__ = lambda self, other: add(other, self)
Specification.isequal = isequal
Specification.inverse = inverse
Specification.isempty = isempty
Specification.project = project
Specification.check = lambda self, S, *args: check(self, S, *args)  # Return full tuple (res, indSpec, indObj)
Specification.add = add
Specification.splitLogic = splitLogic
Specification.plot = plot
Specification.plotOverTime = plotOverTime
Specification.robustness = robustness
Specification.printSpec = printSpec

__all__ = [
    'Specification', 'create_safety_specification', 'create_invariant_specification', 
    'create_unsafe_specification', 'add', 'check', 'eq', 'isequal', 'inverse', 
    'isempty', 'ne', 'project', 'splitLogic', 'plot', 'plotOverTime', 'robustness', 
    'printSpec'
] 