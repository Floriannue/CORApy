"""
isequal - checks if two linear reset functions are equal up to some tolerance

Syntax:
    res = isequal(linReset1,linReset2)
    res = isequal(linReset1,linReset2,tol)

Inputs:
    linReset1 - linearReset object
    linReset2 - linearReset object
    tol - tolerance (optional)

Outputs:
    res - true/false

Example: 
    A = eye(2); c1 = [1;-1]; c2 = [1;1];
    linReset1 = linearReset(A,c1);
    linReset2 = linearReset(A,c2);
    isequal(linReset1,linReset1);
    isequal(linReset1,linReset2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: nonlinearReset/isequal

Authors:       Mark Wetzlinger
Written:       09-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices


def isequal(linReset1: Any, linReset2: Any, *args, **kwargs) -> bool:
    """
    Checks if two linear reset functions are equal up to some tolerance
    
    Args:
        linReset1: linearReset object
        linReset2: linearReset object or nonlinearReset object
        *args: Optional tolerance argument
        **kwargs: Optional keyword arguments
        
    Returns:
        res: True if equal, False otherwise
    """
    # Default tolerance
    tol = setDefaultValues([np.finfo(float).eps], list(args))[0]
    
    # Check input arguments
    # Redirect to nonlinearReset/isequal if linReset2 is nonlinearReset
    from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
    if isinstance(linReset2, NonlinearReset):
        return linReset2.isequal(linReset1, tol)
    
    # Check number of states and inputs in superclass function
    # This ensures that the matrices are of equal size for further checks below
    # Use AbstractReset.isequal for dimension check (no tolerance needed for dimensions)
    from cora_python.hybridDynamics.abstractReset.isequal import isequal as abstractReset_isequal
    if not abstractReset_isequal(linReset1, linReset2):
        return False
    
    # Compare matrices A, B, and vector c
    res = (compareMatrices(linReset1.A, linReset2.A, tol, "equal", True) and
           compareMatrices(linReset1.B, linReset2.B, tol, "equal", True) and
           compareMatrices(linReset1.c, linReset2.c, tol, "equal", True))
    
    return res

