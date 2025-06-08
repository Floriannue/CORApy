"""
display - Displays the properties of a zonotope object (center and
    generator matrix) on the command window

Syntax:
    display(Z)

Inputs:
    Z - zonotope object

Outputs:
    ---

Example:
    Z = zonotope([1, 0], [[1, 2, -1], [0, -1, 1]])
    display(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       14-September-2006 (MATLAB)
Last update:   09-June-2020 (MATLAB)
Python translation: 2025
"""

import numpy as np


def display(Z):
    """
    Displays the properties of a zonotope object
    
    Args:
        Z: zonotope object
    """
    # Check for special cases
    if Z.is_empty():
        print(f"Empty zonotope in R^{Z.dim()}")
        return
    
    # Display basic information
    print(f"Zonotope in R^{Z.dim()}")
    print()
    
    # Display center
    print("c:")
    print(Z.c)
    print()
    
    # Display generators (limit display for large matrices)
    if Z.G.size > 0:
        print("G:")
        # Limit display to reasonable size (similar to MATLAB's DISPLAYDIM_MAX)
        max_display = 10
        if Z.G.shape[1] <= max_display:
            print(Z.G)
        else:
            print(f"{Z.G[:, :max_display]} ... ({Z.G.shape[1]} generators total)")
    else:
        print("G: (no generators)")
    print() 