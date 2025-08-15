"""
eventFcn - event function for polytopes (exact MATLAB translation)

Returns (val, isterminal, direction) for ODE event handling.
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def eventFcn(P, x: np.ndarray, direction_val: int = 1):
    # Ensure constraints
    P.constraints()

    # No equalities
    if P.be.size == 0:
        # val = A*x - b
        val = (P.A @ x.reshape(-1, 1)) - P.b
        isterminal = np.ones((P.b.shape[0], 1))
        direction = np.ones((P.b.shape[0], 1)) * direction_val
        return val.flatten(), isterminal.flatten(), direction.flatten()

    # Constrained hyperplane
    if P.representsa_('conHyperplane', 1e-12):
        val = (P.Ae @ x.reshape(-1, 1)) - P.be
        isterminal = np.array([1])
        # MATLAB leaves direction unspecified; mirror input scalar
        direction = np.array([direction_val])
        return val.flatten(), isterminal.flatten(), direction.flatten()

    raise CORAerror('CORA:notSupported', 'Given polytope type not supported.')


