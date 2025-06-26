"""
capsule method for zonotope class
"""

import numpy as np
from .zonotope import Zonotope
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cora_python.contSet.capsule import Capsule

def capsule(Z: Zonotope) -> 'Capsule':
    """
    Encloses a zonotope with a capsule
    
    Args:
        Z: zonotope object
        
    Returns:
        Capsule object that encloses the zonotope
    """
    # Compute orthogonal basis using PCA (SVD)
    G = Z.generators()
    
    # Use SVD on concatenated generators [-G, G] to find principal directions
    combined_G = np.hstack([-G, G])
    B, _, _ = np.linalg.svd(combined_G, full_matrices=True)
    
    # Compute enclosing interval in the transformed space
    Z_transformed = B.T @ Z
    int_transformed = Z_transformed.interval()
    
    # Enclose interval with a capsule
    C_transformed = int_transformed.capsule()
    
    # Back-transformation to original space
    C = B @ C_transformed
    
    return C 