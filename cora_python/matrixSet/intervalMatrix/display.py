"""
display - displays an intervalMatrix object on the command window

Syntax:
    display(intMat)

Inputs:
    intMat - intervalMatrix object

Outputs:
    str_repr - string representation

Example:
    intMat = IntervalMatrix(np.array([[1, 2, 3], [2, 3, 1]]), np.array([[1, 0, 2], [0, 1, 1]]))
    display_str = intMat.display()

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       18-June-2010 (MATLAB)
Last update:   03-April-2023 (MW, add empty case)
               25-April-2024 (TL, harmonized display with contSet classes)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def display_(intMat: 'IntervalMatrix') -> str:
    """
    Displays an intervalMatrix object (internal function that returns string)
    
    Args:
        intMat: intervalMatrix object
        
    Returns:
        str_repr: string representation
    """
    
    if intMat.isempty():
        return f"IntervalMatrix (empty matrix)"
    
    # Build the string representation
    lines = []
    lines.append(f"IntervalMatrix:")
    lines.append(f"- dimension: {intMat.dim()}")
    lines.append("")
    
    # Display the interval representation
    # Use the interval's display_ method
    from cora_python.contSet.interval.display import display_ as interval_display_
    interval_display = interval_display_(intMat.int)
    lines.append(interval_display)
    
    return "\n".join(lines)


def display(intMat: 'IntervalMatrix') -> None:
    """
    Displays an intervalMatrix object (prints to stdout)
    
    Args:
        intMat: intervalMatrix object
    """
    print(display_(intMat), end='') 