"""
volume_ - computes the volume of a set (internal use)

This function provides the internal implementation for volume computation.
It should be overridden in subclasses to provide specific volume computation logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def volume_(S: 'ContSet', method: str = 'exact', order: int = 5) -> float:
    """
    Computes the volume of a set (internal use, see also contSet/volume)
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific volume computation logic.
    
    Args:
        S: contSet object
        method: Method for volume computation
        order: Order parameter for computation
        
    Returns:
        float: Volume of the set
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S = interval([1, 2], [3, 4])
        >>> vol = volume_(S, 'exact', 5)
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'volume_ not implemented for {type(S).__name__} with method {method}') 