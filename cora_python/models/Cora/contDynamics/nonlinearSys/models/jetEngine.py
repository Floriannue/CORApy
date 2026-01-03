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
    jetEngine - dynamic equation for jet engine model
    
    Syntax:
        dx = jetEngine(x, u)
    
    Inputs:
        x - state vector (2x1)
        u - input vector (1x1)
    
    Outputs:
        dx - derivative of state vector (2x1)
    
    Example:
        x = np.array([[1], [0]])
        u = np.array([[0.5]])
        dx = jetEngine(x, u)
    """
    # MATLAB: dx = [x(2); -x(1) - x(2) + u(1)];
    # Handle both array and matrix inputs
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        dx = np.array([[x[1, 0]], [-x[0, 0] - x[1, 0] + u[0, 0]]])
    else:
        # Fallback for other types
        dx = np.array([[x[1]], [-x[0] - x[1] + u[0]]])
    return dx


