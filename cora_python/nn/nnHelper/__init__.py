"""
nnHelper module for neural network helper functions
"""

from .validateNNoptions import validateNNoptions
from .leastSquarePolyFunc import leastSquarePolyFunc
from .leastSquareRidgePolyFunc import leastSquareRidgePolyFunc
from .minMaxDiffOrder import minMaxDiffOrder
from .getDerInterval import getDerInterval
from .minMaxDiffPoly import minMaxDiffPoly
from .getOrderIndicesG import getOrderIndicesG
from .getOrderIndicesGI import getOrderIndicesGI
from .calcSquared import calcSquared
from .calcSquaredG import calcSquaredG
from .calcSquaredE import calcSquaredE
from .calcSquaredGInd import calcSquaredGInd
from .findBernsteinPoly import findBernsteinPoly
from .calcAlternatingDerCoeffs import calcAlternatingDerCoeffs
from .lookupDf import lookupDf, clear_lookup_table, get_lookup_table_info
from .compBoundsPolyZono import compBoundsPolyZono
from .conversionConZonoStarSet import conversionConZonoStarSet
from .conversionStarSetConZono import conversionStarSetConZono
from .heap import Heap
from .reducePolyZono import reducePolyZono
from .restructurePolyZono import restructurePolyZono
from .setDefaultFields import setDefaultFields
from .validateRLoptions import validateRLoptions

__all__ = [
    'leastSquarePolyFunc',
    'leastSquareRidgePolyFunc', 
    'minMaxDiffOrder',
    'getDerInterval',
    'minMaxDiffPoly',
    'validateNNoptions',
    'getOrderIndicesG',
    'getOrderIndicesGI',
    'calcSquared',
    'calcSquaredG',
    'calcSquaredE',
    'calcSquaredGInd',
    'findBernsteinPoly',
    'calcAlternatingDerCoeffs',
    'lookupDf',
    'clear_lookup_table',
    'get_lookup_table_info',
    'compBoundsPolyZono',
    'conversionConZonoStarSet',
    'conversionStarSetConZono',
    'Heap',
    'reducePolyZono',
    'restructurePolyZono',
    'setDefaultFields',
    'validateRLoptions'
]
