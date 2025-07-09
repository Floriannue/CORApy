"""
tensorMultiplication_zono method for zonotope class
"""

import numpy as np
from typing import Dict, Any
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def tensorMultiplication_zono(Z: Zonotope, M, options: Dict[str, Any]) -> Zonotope:
    """
    Computes {M_{ijk...l}*x_j*x_k*...*x_l|x \in Z} when the center of Z is the origin 
    and M is a matrix zonotope
    
    Args:
        Z: zonotope object
        M: matrix zonotope object
        options: options dictionary
        
    Returns:
        Zonotope object
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        # M should be a matrix zonotope object
        Zres = tensorMultiplication_zono(Z, M, options)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Compute zonotope of center
    from .tensorMultiplication import tensorMultiplication
    Zres = tensorMultiplication(Z, M.center, options)
    
    # Add results from generators
    from .plus import plus
    for i in range(M.numgens):
        Zres = plus(Zres, tensorMultiplication(Z, M.generator[i], options))
    
    return Zres 