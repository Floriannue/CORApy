"""
capsule - encloses a zonotope with a capsule

Syntax:
    C = capsule(Z)

Inputs:
    Z - zonotope object

Outputs:
    C - capsule object

Example:
    from cora_python.contSet.zonotope import Zonotope, capsule
    import numpy as np
    Z = Zonotope(np.array([[1], [-1]]), np.array([[2, -3, 1], [0.5, 1, -2]]))
    C = capsule(Z)

Other m-files required: interval (constructor)
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       23-December-2019 (MATLAB)
Last update:   --- (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""
import numpy as np
from .zonotope import Zonotope
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cora_python.contSet.capsule import Capsule

def capsule(Z: Zonotope) -> 'Capsule':
    """
    Encloses a zonotope with a capsule.
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