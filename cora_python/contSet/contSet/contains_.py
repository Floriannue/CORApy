"""
contains_ - determines if a set contains a set or a point (internal method)

Syntax:
    [res,cert,scaling] = contains_(S1,S2,method,tol,maxEval,certToggle,scalingToggle)

Inputs:
    S1 - contSet object
    S2 - contSet object or numerical vector
    method - method used for the containment check
    tol - tolerance for the containment check
    maxEval - maximum number of evaluations
    certToggle - if set to True, cert will be computed
    scalingToggle - if set to True, scaling will be computed

Outputs:
    res - true/false
    cert - certificate
    scaling - scaling factor

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/contains

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       22-March-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union, Tuple
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def contains_(S1: 'ContSet', S2: Union['ContSet', np.ndarray], method='exact', tol=1e-12, maxEval=200, cert_toggle=False, scaling_toggle=False):
    """
    Determines if a set contains a set or a point (internal method)
    
    This function uses polymorphic dispatch to call the appropriate subclass
    implementation of contains_, or provides the base implementation.
    
    Args:
        S1: contSet object
        S2: contSet object or numerical vector
        method: method used for the containment check
        tol: tolerance for the containment check
        maxEval: maximum number of evaluations
        cert_toggle: if True, cert will be computed
        scaling_toggle: if True, scaling will be computed
        
    Returns:
        tuple: (res, cert, scaling) where:
            - res: True/False
            - cert: certificate
            - scaling: scaling factor
    """
    # Check if subclass has overridden contains_ method
    base_class = type(S1).__bases__[0] if type(S1).__bases__ else None
    if (hasattr(type(S1), 'contains_') and 
        base_class and hasattr(base_class, 'contains_') and
        type(S1).contains_ is not base_class.contains_):
        return type(S1).contains_(S2, method, tol, maxEval, cert_toggle, scaling_toggle)
    else:
        # Base implementation - throw error as this method should be overridden
        raise CORAerror("CORA:noops", f"Function contains_ not implemented for class {type(S1).__name__}") 