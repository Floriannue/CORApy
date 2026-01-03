"""
isequal - checks if two nonlinear reset functions, or a nonlinear reset
   function and a linear reset function, are equal up to some tolerance

Syntax:
    res = isequal(nonlinReset1,nonlinReset2)
    res = isequal(nonlinReset1,nonlinReset2,tol)

Inputs:
    nonlinReset1 - nonlinearReset object, linearReset object
    nonlinReset2 - nonlinearReset object, linearReset object
    tol - tolerance (optional)

Outputs:
    res - true/false

Example: 
    f = @(x,u) [-x(1)*x(2); x(2) - u(1)];
    g = @(x,u) [-x(1)*x(2); x(1) - u(1)];
    nonlinReset1 = nonlinearReset(f);
    nonlinReset2 = nonlinearReset(g);
    isequal(nonlinReset1,nonlinReset1);
    isequal(nonlinReset1,nonlinReset2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: linearReset/isequal, isequalFunctionHandle

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
from cora_python.g.functions.matlab.function_handle.isequalFunctionHandle import isequalFunctionHandle


def isequal(nonlinReset1: Any, nonlinReset2: Any, *args, **kwargs) -> bool:
    """
    Checks if two nonlinear reset functions, or a nonlinear reset function and a linear reset function, are equal up to some tolerance
    
    Args:
        nonlinReset1: nonlinearReset object or linearReset object
        nonlinReset2: nonlinearReset object or linearReset object
        *args: Optional tolerance argument
        **kwargs: Optional keyword arguments
        
    Returns:
        res: True if equal, False otherwise
    """
    # Default tolerance
    tol = setDefaultValues([np.finfo(float).eps], list(args))[0]
    
    # Check input arguments
    # MATLAB: inputArgsCheck({{nonlinReset1,'att','nonlinearReset'};...})
    # For now, we'll skip strict type checking to allow flexibility
    
    # Check number of states and inputs in superclass function
    from cora_python.hybridDynamics.abstractReset.isequal import isequal as abstractReset_isequal
    if not abstractReset_isequal(nonlinReset1, nonlinReset2):
        return False
    
    # Convert to second input argument to nonlinearReset object
    from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
    if isinstance(nonlinReset2, LinearReset):
        # Convert LinearReset to NonlinearReset
        nonlinReset2 = nonlinReset2.nonlinearReset()
    
    # Compare function handles
    res = isequalFunctionHandle(nonlinReset1.f, nonlinReset2.f)
    
    return res

