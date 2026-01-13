"""
Private functions for linearParamSys operations
"""

from .priv_mappingMatrix import priv_mappingMatrix
from .priv_highOrderMappingMatrix import priv_highOrderMappingMatrix
from .priv_tie import priv_tie
from .priv_inputTie import priv_inputTie
from .priv_inputSolution import priv_inputSolution
from .priv_dependentHomSol import priv_dependentHomSol

__all__ = [
    'priv_mappingMatrix',
    'priv_highOrderMappingMatrix',
    'priv_tie',
    'priv_inputTie',
    'priv_inputSolution',
    'priv_dependentHomSol'
]
