"""
deleteZeros - (DEPRECATED -> compact)

Syntax:
    Z = deleteZeros(Z)

Inputs:
    Z - zonotope object

Outputs:
    Z - zonotope object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/compact_

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 15-January-2009 (MATLAB)
Last update: 29-July-2023 (MW, merged into compact) (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning


def deleteZeros(Z: Zonotope) -> Zonotope:
    """
    Delete zero generators (DEPRECATED -> use compact instead)
    
    Args:
        Z: zonotope object
        
    Returns:
        Zonotope object with zero generators removed
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 0]]))
        Z = deleteZeros(Z)
    """
    # Show deprecation warning
    CORAwarning("CORA:deprecated", 'function', 'zonotope/deleteZeros', 'CORA v2024',
                'When updating the code, please replace every function call ''deleteZeros(Z)'' with ''compact(Z,''zeros'')''.',
                'This change was made in an effort to unify the syntax across all set representations.')
    
    # Call compact with zeros option
    from .compact_ import compact_
    Z = compact_(Z, 'zeros', float(np.finfo(float).eps))
    
    return Z 