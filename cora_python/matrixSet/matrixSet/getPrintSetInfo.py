"""
getPrintSetInfo - returns all information to properly print a set 
   to the command window 

Syntax:
   [abbrev,propertyOrder] = getPrintSetInfo(S)

Inputs:
   S - contSet or matrixSet

Outputs:
   abbrev - set abbreviation
   propertyOrder - order of the properties

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: printSet

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       10-October-2024 (MATLAB)
Python translation: 2025
"""

from typing import Tuple, List, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .matrixSet import MatrixSet


def getPrintSetInfo(S: 'MatrixSet') -> Tuple[str, List[str]]:
    """
    Returns all information to properly print a set to the command window
    
    Args:
        S: matrixSet object
        
    Returns:
        abbrev: set abbreviation
        propertyOrder: order of the properties
        
    Raises:
        CORAerror: This is an abstract method that should be overridden
    """
    
    raise CORAerror('CORA:noops', S) 