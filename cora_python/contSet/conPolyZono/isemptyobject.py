"""
isemptyobject - checks whether a constrained polynomial zonotope contains
   any information at all; consequently, the set is interpreted as the
   empty set 

Syntax:
   res = isemptyobject(cPZ)

Inputs:
   cPZ - conPolyZono object

Outputs:
   res - true/false

Authors:       Mark Wetzlinger (MATLAB)
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conPolyZono import ConPolyZono

def isemptyobject(cPZ: 'ConPolyZono') -> bool:
    """
    Checks whether a constrained polynomial zonotope contains any information at all.

    Args:
        cPZ: ConPolyZono object.

    Returns:
        True if the object is empty, False otherwise.
    """
    # In MATLAB, this checks if all properties are empty. Replicate that logic.
    # MATLAB: isnumeric(cPZ.c) && isempty(cPZ.c) ...
    # In Python, for NumPy arrays, .size == 0 checks if it's empty.
    return (cPZ.c.size == 0 and
            cPZ.G.size == 0 and
            cPZ.E.size == 0 and
            cPZ.A.size == 0 and
            cPZ.b.size == 0 and
            cPZ.EC.size == 0 and
            cPZ.GI.size == 0 and
            cPZ.id.size == 0)
