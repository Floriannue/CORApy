"""
minnorm - computes the point whose norm is minimal with respect to the
    center of the zonotope; caution: this function requires the halfspace
    representation of the zonotope and thus scales exponentially with
    respect to the dimension

Syntax:
    val, x = minnorm(Z)

Inputs:
    Z - zonotope object

Outputs:
    val - value of minimum zonotope norm, i.e., the point on the
          boundary of Z which has minimum distance to the zonotope center
    x - point on boundary attaining minimum norm

Example: 
    Z = Zonotope(np.array([[2], [1]]), np.array([[1, -1, 0], [1, 2, 3]]))
    val, x = minnorm(Z)

References:
    [1] S. Boyd and L. Vandenberghe, "Convex optimization", Cambridge
        University Press, 2004

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: norm

Authors:       Victor Gassmann
Written:       18-September-2019
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .zonotope import Zonotope


def minnorm(Z: 'Zonotope') -> Tuple[float, np.ndarray]:
    """
    Computes the point whose norm is minimal with respect to the center of the zonotope
    
    Args:
        Z: zonotope object
        
    Returns:
        val: minimum norm value
        x: point on boundary attaining minimum norm
    """
    # Import here to avoid circular import
    from .polytope import polytope
    
    # Get halfspace representation
    # P = polytope(Z - Z.c)
    Z_centered = Z - Z.c
    P = polytope(Z_centered)
    
    A = P.A
    b = P.b
    
    # Compute min norm (obtained by rewriting OP in [1], Sec. 8.4.2, using
    # ||a_i||_2 = 1 and argmin -log(det(scalarVar))=argmax scalarVar
    b_squared = b**2
    ind = np.argmin(b_squared)
    val_squared = b_squared[ind]
    
    # Compute the point on the boundary
    # MATLAB: x = A(ind,:)'*b(ind) + Z.c
    # This is matrix-vector multiplication: A[ind, :].T @ b[ind] + Z.c
    x = A[ind, :].reshape(-1, 1) * b[ind] + Z.c
    val = float(np.sqrt(val_squared))
    
    return val, x 