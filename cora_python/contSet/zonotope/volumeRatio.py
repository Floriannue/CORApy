"""
volumeRatio - computes the approximate volume ratio of a zonotope and its over-approximating polytope

Syntax:
    ratio = volumeRatio(Z, P, dims)

Inputs:
    Z - zonotope object
    P - polytope object
    dims - considered dimensions for the approximation (optional)

Outputs:
    ratio - approximated normalized volume ratio

Example:
    from cora_python.contSet.zonotope import Zonotope, volumeRatio
    import numpy as np
    Z = Zonotope(np.array([[1], [0]]), np.random.rand(2, 5))
    P = polytope(Z)
    ratio = volumeRatio(Z, P, 1)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       12-September-2008 (MATLAB)
Last update:   28-August-2019 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""
import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def volumeRatio(Z: Zonotope, P, dims: Optional[int] = None) -> float:
    """
    Computes the approximate volume ratio of a zonotope and its over-approximating polytope.
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Write inputs to variables
    if dims is None:
        dims = Z.dim()  # Use dim() method instead of Z.c.shape[0]
    
    # Obtain dimension
    n = Z.dim()
    
    # Generate dim vector
    dimVector = np.arange(1, dims + 1)
    
    # Obtain number of iterations
    iterations = n - dims + 1
    
    # Init projected zonotope
    Zproj = Z
    
    partialRatio = np.zeros(iterations)
    
    for i in range(iterations):
        # Projected dimensions
        projDims = dimVector + i - 1
        
        # Project zonotope
        from .project import project
        Zproj = project(Z, projDims)
        
        # Project polytope
        Pproj = project(P, projDims)
        
        # Compute volume of the projected zonotope and polytope
        from .volume_ import volume_
        volZ = volume_(Zproj, 'exact')
        volP = volume_(Pproj, 'exact')
        
        # Obtain normalized ratio
        if volZ > 0:
            partialRatio[i] = (volP / volZ) ** (1 / dims)
        else:
            partialRatio[i] = 0
    
    # Final ratio is the mean value of the partial ratios
    ratio = np.mean(partialRatio)
    
    return ratio 