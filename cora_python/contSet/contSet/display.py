"""
display - Displays the properties of a contSet object on the command window

This function displays basic information about a contSet object, including
its class name and dimension.

Authors: Matthias Althoff, Victor Gassmann (MATLAB)
         Python translation by AI Assistant
Written: 02-May-2007 (MATLAB)
Last update: 26-October-2023 (MATLAB)
Python translation: 2025
"""

from .dim import dim


def display(S: 'ContSet') -> None:
    """
    Displays the properties of a contSet object (dimension) on the command window
    
    Args:
        S: contSet object to display
        
    Returns:
        None
        
    Example:
        >>> S = interval([1, 2], [3, 4])
        >>> display(S)
        Interval:
        - dimension: 2
    """
    # Display class name and dimension
    print(f"{type(S).__name__}:")
    print(f"- dimension: {dim(S)}") 