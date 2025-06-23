"""
enclosePoints - enclose a point cloud with a set

This is the base class implementation that throws an error.
Each subclass should override this method with its own implementation.

Syntax:
    S = ContSet.enclosePoints(points)
    S = ContSet.enclosePoints(points, method)

Inputs:
    points - matrix storing point cloud (dimension: [n,p] for p points)
    method - (optional) method

Outputs:
    S - ContSet object

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def enclosePoints(*args, **kwargs):
    """
    enclosePoints - enclose a point cloud with a set
    
    This is the base implementation that throws an error since it should
    be overridden in subclasses.
    
    Args:
        *args: Variable arguments including points and optional method
        **kwargs: Keyword arguments
        
    Raises:
        CORAerror: This method is not implemented for the base ContSet class
    """
    # Throw error since this should be overridden in subclass
    raise CORAerror("CORA:noops", *args) 