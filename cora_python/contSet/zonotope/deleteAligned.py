"""
deleteAligned method for zonotope class (DEPRECATED)
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning


def deleteAligned(Z: Zonotope) -> Zonotope:
    """
    Delete aligned generators (DEPRECATED -> use compact instead)
    
    Args:
        Z: zonotope object
        
    Returns:
        Zonotope object with aligned generators removed
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.array([[1, 1], [0, 0]]))
        Z = deleteAligned(Z)
    """
    # Show deprecation warning
    CORAwarning("CORA:deprecated", 'function', 'zonotope/deleteAligned', 'CORA v2024',
                'When updating the code, please replace every function call ''deleteAligned(Z)'' with ''compact(Z,''aligned'')''.',
                'This change was made in an effort to unify the syntax across all set representations.')
    
    # Call compact with aligned option
    from .compact_ import compact_
    Z = compact_(Z, 'aligned', float(np.finfo(float).eps))
    
    return Z 