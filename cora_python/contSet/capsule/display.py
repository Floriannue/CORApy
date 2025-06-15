"""
display - Displays the properties of a capsule object (center, generator,
   radius) on the command window

Syntax:
   display(C)

Inputs:
   C - capsule

Outputs:
   ---

Example: 
   C = capsule([1;1],[1;0],0.5);
   display(C);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 04-March-2019 (MATLAB)
Last update: 02-May-2020 (MW, add empty case)
Python translation: 2025
"""

import numpy as np


def display(C) -> None:
    """
    Displays the properties of a capsule object
    
    Args:
        C: capsule object
    """
    print()
    
    # Check for empty capsule
    if C.is_empty():
        print(f"Empty capsule in R^{C.dim()}")
        print()
        return
    
    print("capsule")
    print()
    
    # Display dimension
    print(f"dimension: {C.dim()}")
    print()
    
    # Display center
    print("center:")
    if C.c is not None:
        print(C.c.flatten())
    else:
        print("[]")
    print()
    
    # Display generator
    print("generator:")
    if C.g is not None:
        print(C.g.flatten())
    else:
        print("[]")
    print()
    
    # Display radius
    print("radius:")
    print(C.r)
    print() 