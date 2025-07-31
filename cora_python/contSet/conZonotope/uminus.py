"""
uminus - overloaded unary '-' operator for conZonotope objects

Syntax:
    cZ = -cZ1
    cZ = uminus(cZ1)

Inputs:
    cZ1 - conZonotope object

Outputs:
    cZ - resulting conZonotope object

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 22-March-2007 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conZonotope import ConZonotope


def uminus(self: 'ConZonotope') -> 'ConZonotope':
    """
    Overloaded unary '-' operator for conZonotope objects
    
    Args:
        self: conZonotope object to negate
        
    Returns:
        ConZonotope: Negated conZonotope object (-self)
    """
    from .conZonotope import ConZonotope
    
    # Handle empty conZonotope case
    if self.isemptyobject():
        return ConZonotope.empty(self.dim())
    
    # Negate center and generators
    c = -self.c
    G = -self.G if self.G.size > 0 else self.G
    
    # Keep constraints the same (A and b)
    A = self.A
    b = self.b
    
    return ConZonotope(c, G, A, b) 