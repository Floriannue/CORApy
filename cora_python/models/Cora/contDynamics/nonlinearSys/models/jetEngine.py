"""
jetEngine - dynamic equation for jet engine model

This is a Python translation of the MATLAB CORA implementation.

Source: cora_matlab/models/Cora/contDynamics/nonlinearSys/models/jetEngine.m

Authors: MATLAB: (Original authors)
         Python: AI Assistant
"""

import numpy as np


def jetEngine(x, u):
    """
    jetEngine - jet engine model from [1], Example 7
    
    Syntax:
        dx = jetEngine(x, u)
    
    Inputs:
        x - state vector (2x1)
        u - input vector (not used in this model)
    
    Outputs:
        dx - time-derivate of the system state (2x1)
    
    References:
        [1] Under-approximate Flowpipes for Non-linear Continuous Systems
            Xin Chen, Sriram Sankaranarayanan, Erika Abraham
            ISBN: 978-0-9835678-4-4.
    
    Example:
        x = np.array([[1], [0]])
        u = np.array([[0.5]])
        dx = jetEngine(x, u)
    """
    # MATLAB: dx(1,1) = -x(2) - 1.5*x(1)^2 - 0.5*x(1)^3 - 0.5;
    # MATLAB: dx(2,1) = 3*x(1) - x(2);
    # Handle both array and matrix inputs
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        x1 = x[0, 0] if x.shape[0] > 0 else x[0]
        x2 = x[1, 0] if x.shape[0] > 1 else x[1]
        
        # dx(1) = -x(2) - 1.5*x(1)^2 - 0.5*x(1)^3 - 0.5
        dx1 = -x2 - 1.5 * x1**2 - 0.5 * x1**3 - 0.5
        # dx(2) = 3*x(1) - x(2)
        dx2 = 3 * x1 - x2
        
        dx = np.array([[dx1], [dx2]])
    else:
        # Fallback for other types
        dx1 = -x[1] - 1.5 * x[0]**2 - 0.5 * x[0]**3 - 0.5
        dx2 = 3 * x[0] - x[1]
        dx = np.array([[dx1], [dx2]])
    return dx


