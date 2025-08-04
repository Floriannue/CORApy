"""
 center - returns the center of a zonotope

 Syntax:
     c = center(Z)

 Inputs:
     Z - zonotope object

 Outputs:
     c - center of the zonotope Z

 Example:
     Z = zonotope([1;0],[1 0; 0 1]);
     c = center(Z)

 Other m-files required: none
 Subfunctions: none
 MAT-files required: none

 See also: none

 Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
                Python translation by AI Assistant
 Written:       30-September-2006 (MATLAB)
 Last update:   14-March-2021 (MW, empty set) (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
 """


import numpy as np
from .zonotope import Zonotope


def center(Z: Zonotope) -> np.ndarray:

    return Z.c 