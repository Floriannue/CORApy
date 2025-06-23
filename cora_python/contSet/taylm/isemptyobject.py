"""
isemptyobject - checks whether a Taylor model contains any information

Syntax:
    res = isemptyobject(t)

Inputs:
    t - taylm object

Outputs:
    res - true/false
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.taylm.taylm import Taylm


def isemptyobject(t: 'Taylm') -> bool:
    """Checks whether a Taylor model contains any information"""
    
    # Check if monomials exist and have content
    if not hasattr(t, 'monomials') or t.monomials.size == 0:
        return True
    
    # Check dimension
    return t.dim() == 0 