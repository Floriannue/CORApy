"""
getfcn - returns the function handle of the continuous function specified
   by the nonlinear system object

Syntax:
    han = getfcn(nlnsys, params)

Inputs:
    nlnsys - nonlinearSys object
    params - model parameters (must contain 'u')

Outputs:
    han - function handle

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       17-October-2007 (MATLAB)
Python translation: 2025
"""

from typing import Any, Dict, Callable
import numpy as np


def getfcn(nlnsys: Any, params: Dict[str, Any]) -> Callable:
    """
    Return a callable f(t, x) for ODE integration.
    """
    u = params.get('u', None)

    def f(_t, x):
        x_vec = np.asarray(x).reshape(-1, 1)
        if u is None:
            u_vec = np.zeros((nlnsys.nr_of_inputs, 1))
        else:
            u_vec = np.asarray(u).reshape(-1, 1)
        dx = nlnsys.mFile(x_vec, u_vec)
        return np.asarray(dx).flatten()

    return f
