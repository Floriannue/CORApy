"""
evaluate - evaluates the linear reset function for a given state (set)
   and input (set)

Syntax:
    x_ = evaluate(linReset,x)
    x_ = evaluate(linReset,x,u)

Inputs:
    linReset - linearReset object
    x - state before reset
    u - (optional) input before reset

Outputs:
    x_ - state after reset

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: nonlinearReset/evaluate

Authors:       Mark Wetzlinger
Written:       07-September-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Optional
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues


def evaluate(linReset: Any, x: Any, u: Optional[Any] = None) -> Any:
    """
    Evaluates the linear reset function for a given state (set) and input (set)
    
    Args:
        linReset: linearReset object
        x: state before reset
        u: (optional) input before reset
        
    Returns:
        x_: state after reset (x_ = A*x + B*u + c)
    """
    # Set default value
    # MATLAB: u = setDefaultValues({zeros(linReset.inputDim,1)},varargin);
    if u is None:
        u = np.zeros((linReset.inputDim, 1))
    
    # Evaluate reset function
    # MATLAB: x_ = linReset.A * x + linReset.B * u + linReset.c;
    # MATLAB always has A, B, c as non-empty matrices (defaulted to zeros in constructor)
    # So we can directly compute without None checks
    x_ = linReset.A @ x + linReset.B @ u + linReset.c
    
    return x_


