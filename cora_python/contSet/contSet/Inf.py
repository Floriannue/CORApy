"""
Inf - instantiates a fullspace set

This static method creates a fullspace set of the specified dimension.
It should be overridden in subclasses that support fullspace set instantiation.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 09-January-2024 (MATLAB)
Python translation: 2025
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def Inf(n: int) -> 'ContSet':
    """
    Instantiates a fullspace set
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific fullspace set instantiation.
    
    Args:
        n: Dimension of the fullspace set
        
    Returns:
        ContSet: Fullspace set object
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes like interval, fullspace, etc.
        >>> S = interval.Inf(2)  # Creates 2D fullspace interval
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:notSupported',
                   'The chosen subclass of contSet does not support a fullspace set instantiation.') 