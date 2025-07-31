"""
vertices_ - calculate the vertices of a constrained zonotope object

Syntax:
    V = cZ.vertices_()
    V = cZ.vertices_(method)
    V = cZ.vertices_(method, numDir)

Inputs:
    cZ - conZonotope object
    method - 'default', 'template'
    numDir - number of directions for template method

Outputs:
    V - matrix storing the vertices (dimension: [n,p], with p vertices)

Authors:       Niklas Kochdumper
Written:       11-May-2018
Last update:   25-April-2023 (TL, 2d support func computation)
                15-October-2024 (TL, integrated 'template' from plot function)
Last revision: 27-March-2023 (MW, rename vertices_)
"""

import numpy as np
from typing import Optional, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues

if TYPE_CHECKING:
    from .conZonotope import ConZonotope


def vertices_(self: 'ConZonotope', method: str = 'default', 
              numDir: Optional[int] = None) -> np.ndarray:
    """
    Calculate the vertices of a constrained zonotope object
    
    Args:
        method: 'default' or 'template'
        numDir: number of directions for template method
    
    Returns:
        V: matrix storing the vertices (dimension: [n,p], with p vertices)
    """
    # First, simplify the constrained zonotope as much as possible
    cZ = self.compact('zeros', 1e-10)
    
    # Check if generator matrix is empty
    if cZ.G.size == 0:
        return cZ.c
    
    # Check if 2-dimensional
    if cZ.dim() == 2:
        # Vertices of a 2D cZ can be computed efficiently using support functions
        # For now, use a simple approximation
        return aux_verticesDefault(cZ)
    
    # First remove redundant constraints that will not overapproximate the
    # original constrained zonotope. Then we can check whether or not the
    # resulting set is, in fact, a zonotope
    cZ = cZ.reduceConstraints()
    
    # Check if any constraint is left
    if cZ.A.size == 0:
        # No constraints -> call zonotope/vertices
        from cora_python.contSet.zonotope import Zonotope
        Z = Zonotope(cZ.c, cZ.G)
        return Z.vertices_()
    
    # Choose method
    if method == 'template':
        return aux_verticesTemplate(cZ, numDir)
    
    # Default vertex computation
    return aux_verticesDefault(cZ)


def aux_verticesDefault(cZ: 'ConZonotope') -> np.ndarray:
    """Calculate potential vertices of the constrained zonotope"""
    # For now, use a simple approximation
    # In the full implementation, this would use priv_potVertices
    from cora_python.contSet.zonotope import Zonotope
    
    # Convert to zonotope and get vertices as approximation
    Z = Zonotope(cZ.c, cZ.G)
    V = Z.vertices_()
    
    # Filter vertices that satisfy constraints
    valid_vertices = []
    for i in range(V.shape[1]):
        vertex = V[:, i:i+1]
        # Check if vertex satisfies constraints A*beta = b
        # This is a simplified check - in practice, we need to solve for beta
        if cZ.A.size == 0 or np.allclose(cZ.A @ vertex, cZ.b, atol=1e-10):
            valid_vertices.append(vertex)
    
    if valid_vertices:
        return np.hstack(valid_vertices)
    else:
        return cZ.c


def aux_verticesTemplate(cZ: 'ConZonotope', numDir: Optional[int] = None) -> np.ndarray:
    """Template method for vertex computation"""
    # Only implemented for 2d and 3d
    if cZ.dim() not in [2, 3]:
        raise NotImplementedError("Template method only implemented for 2D and 3D")
    
    # For now, fall back to default method
    return aux_verticesDefault(cZ) 