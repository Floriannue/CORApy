"""
spectraShadow - Converts a zonotope to a spectrahedral shadow

Syntax:
    SpS = spectraShadow(Z)

Inputs:
    Z - zonotope object

Outputs:
    SpS - spectraShadow object

Example:
    from cora_python.contSet.zonotope import Zonotope, spectraShadow
    import numpy as np
    Z = Zonotope(np.array([[0], [0]]), np.eye(2))
    SpS = spectraShadow(Z)
    # SpS is a spectraShadow object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       01-August-2023 (MATLAB)
Last update:   --- (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""
import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def spectraShadow(Z: Zonotope):
    """
    Converts a zonotope to a spectrahedral shadow.
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
    SpS.emptySet.val = Z.representsa_('emptySet', 1e-10)
    SpS.fullDim.val, _ = Z.isFullDim()
    SpS.center.val = Z.center()
    
    return SpS 