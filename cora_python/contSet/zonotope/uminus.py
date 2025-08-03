"""
uminus - overloaded unary '-' operator for zonotope objects

Syntax:
    Z = -Z1
    Z = uminus(Z1)

Inputs:
    Z1 - zonotope object

Outputs:
    Z - resulting zonotope object

Example:
    from cora_python.contSet.zonotope import Zonotope, uminus
    import numpy as np
    Z1 = Zonotope(np.array([[1], [0]]), np.array([[1, 0], [0, 1]]))
    Z = uminus(Z1)  # or Z = -Z1

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2006 (MATLAB)
Last update:   22-March-2007 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""
import numpy as np
from .zonotope import Zonotope

def uminus(Z: Zonotope) -> Zonotope:
    """
    Overloaded unary '-' operator for zonotope objects.
    """
    
    # Handle empty zonotope case
    if Z.is_empty():
        return Zonotope.empty(Z.dim())
    
    # Negate center and generators
    c = -Z.c
    G = -Z.G if Z.G.size > 0 else Z.G
    
    return Zonotope(c, G) 