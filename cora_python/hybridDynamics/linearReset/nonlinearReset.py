"""
nonlinearReset - converts a linearReset object into a nonlinearReset object

TRANSLATED FROM: cora_matlab/hybridDynamics/@linearReset/nonlinearReset.m

Syntax:
    nonlinReset = nonlinearReset(linReset)

Inputs:
    linReset - linearReset object

Outputs:
    nonlinReset - nonlinearReset object

Example:
    A = [1 0; 0 1]; B = [1; 0]; c = [-1; 1];
    linReset = LinearReset(A, B, c);
    nonlinReset = NonlinearReset(linReset);

Authors:       Mark Wetzlinger (MATLAB)
Written:       08-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .linearReset import LinearReset


def nonlinearReset(linReset: 'LinearReset') -> 'NonlinearReset':
    """
    Converts a linearReset object into a nonlinearReset object.
    
    Args:
        linReset: linearReset object
    
    Returns:
        NonlinearReset object
    """
    from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
    
    # Empty case
    if linReset.preStateDim == 0:
        f = lambda x, u: np.array([]).reshape(0, 1)
        return NonlinearReset(f)
    
    # Create a function that computes A*x + B*u + c
    # Store matrices in closure
    A = linReset.A
    B = linReset.B if linReset.B is not None and linReset.B.size > 0 else np.zeros((linReset.postStateDim, linReset.inputDim))
    c = linReset.c if linReset.c is not None and linReset.c.size > 0 else np.zeros((linReset.postStateDim, 1))
    
    # Ensure proper shapes
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    
    # Create function handle
    def f(x, u):
        # Ensure x and u are column vectors
        if isinstance(x, (list, tuple)):
            x = np.array(x)
        if isinstance(u, (list, tuple)):
            u = np.array(u)
        
        x = np.atleast_1d(x)
        u = np.atleast_1d(u)
        
        # Ensure column vectors
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        
        # Compute A*x + B*u + c
        result = A @ x + B @ u + c
        
        # Return as column vector
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        return result
    
    # Instantiate nonlinearReset object
    nonlinReset = NonlinearReset(f)
    
    return nonlinReset

