"""
volume_ - computes the volume of a set (internal use)

This function provides the internal implementation for volume computation.
It should be overridden in subclasses to provide specific volume computation logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def volume_(S: 'ContSet', method: str = 'exact') -> float:
    """
    Computes the volume of a set (internal use)
    
    This function uses polymorphic dispatch to call the appropriate subclass
    implementation of volume_, or provides the base implementation.
    
    Args:
        S: contSet object
        method: method for volume computation
        
    Returns:
        float: Volume of the set
        
    Raises:
        CORAerror: If volume_ is not implemented for the specific set type
    """
    # Base implementation - throw error as this method should be overridden
    raise CORAerror("CORA:noops", f"Function volume_ not implemented for class {type(S).__name__}") 