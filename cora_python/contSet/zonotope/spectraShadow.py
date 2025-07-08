"""
spectraShadow method for zonotope class
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def spectraShadow(Z: Zonotope):
    """
    Converts a zonotope to a spectrahedral shadow
    
    Args:
        Z: zonotope object
        
    Returns:
        SpectraShadow object
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.eye(2))
        SpS = spectraShadow(Z)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # The conZonotope implementation is more general, so do it that way
    from cora_python.contSet.conZonotope.conZonotope import ConZonotope
    from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow
    
    # Convert zonotope to conZonotope
    conZ = ConZonotope(Z.c, Z.G, None, None)
    
    # Create spectraShadow from conZonotope
    SpS = SpectraShadow(conZ)
    
    # Additional properties
    SpS.bounded.val = True
    SpS.emptySet.val = _representsa_(Z, 'emptySet', 1e-10)
    SpS.fullDim.val = _isFullDim(Z)
    SpS.center.val = Z.c
    
    return SpS


def _representsa_(Z: Zonotope, type_str: str, tol: float) -> bool:
    """
    Check if zonotope represents a specific type
    """
    if type_str == 'emptySet':
        # Check if zonotope is empty (simplified check)
        if Z.c is None or Z.G is None:
            return True
        # More sophisticated empty set detection would go here
        return False
    return False


def _isFullDim(Z: Zonotope) -> bool:
    """
    Check if zonotope is full dimensional
    """
    if Z.G is None:
        return False
    
    # Check if generators span the full dimension
    rank = np.linalg.matrix_rank(Z.G)
    return rank == Z.G.shape[0] 