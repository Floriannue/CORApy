from .testCase import CoraTestCase
from .validateReach import validateReach
from .sequentialTestCases import sequentialTestCases
from .simResult import to_simResult
from .plot import plot

# Attaching methods to the class
CoraTestCase.validateReach = validateReach
CoraTestCase.sequentialTestCases = sequentialTestCases
CoraTestCase.to_simResult = to_simResult
CoraTestCase.plot = plot

__all__ = ['CoraTestCase'] 