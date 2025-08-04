"""
tensorMultiplication_zono - computes {M_{ijk...l}*x_j*x_k*...*x_l|x \in Z} when the center of Z is the origin and M is a matrix zonotope

Syntax:
    Zres = tensorMultiplication_zono(Z, M, options)

Inputs:
    Z - zonotope object
    M - matrix zonotope object
    options - options dictionary

Outputs:
    Zres - zonotope object

Example:
    from cora_python.contSet.zonotope import Zonotope, tensorMultiplication_zono
    import numpy as np
    Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
    # M should be a matrix zonotope object
    Zres = tensorMultiplication_zono(Z, M, options)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       10-October-2011 (MATLAB)
Last update:   --- (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Dict, Any
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def tensorMultiplication_zono(Z: Zonotope, M, options: Dict[str, Any]) -> Zonotope:
    """
    Computes {M_{ijk...l}*x_j*x_k*...*x_l|x \in Z} when the center of Z is the origin and M is a matrix zonotope.
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