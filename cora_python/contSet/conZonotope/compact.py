"""
compact - compact a constrained zonotope

Syntax:
    cZ = cZ.compact(method, tol)

Inputs:
    method - 'zeros' or 'zeros:full'
    tol - tolerance for zero detection

Outputs:
    cZ - compacted constrained zonotope
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conZonotope import ConZonotope


def compact(self: 'ConZonotope', method: str = 'zeros', tol: float = 1e-10) -> 'ConZonotope':
    """
    Compact a constrained zonotope
    
    Args:
        method: 'zeros' or 'zeros:full'
        tol: tolerance for zero detection
    
    Returns:
        compacted constrained zonotope
    """
    from cora_python.contSet.conZonotope.compact_ import compact_
    return compact_(self, method, tol)