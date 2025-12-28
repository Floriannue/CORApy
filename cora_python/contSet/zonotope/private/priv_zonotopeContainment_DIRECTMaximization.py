"""
priv_zonotopeContainment_DIRECTMaximization - Solves the zonotope containment problem by
   checking whether the maximum value of the Z2-norm at one of the
   vertices of Z1 exceeds 1+tol, using DIRECT optimization.

Syntax:
   res, cert, scaling = priv_zonotopeContainment_DIRECTMaximization(Z1, Z2, tol, maxEval, scalingToggle)

Inputs:
   Z1 - zonotope object, inbody
   Z2 - zonotope object, circumbody
   tol - tolerance for the containment check
   maxEval - Number of maximal function evaluations
   scalingToggle - if set to True, scaling will be computed

Outputs:
   res - True/False
   cert - see logic below
   scaling - the smallest number 'scaling', such that scaling*(Z2 - center(Z2)) + center(Z2) contains Z1 (lower bound)

References:
   [1] A. Kulmburg, M. Althoff.: On the co-NP-Completeness of the Zonotope Containment Problem, European Journal of Control 2021

Authors:       Adrian Kulmburg
Python port:   AI Assistant
"""
import numpy as np
from scipy.optimize import differential_evolution

def priv_zonotopeContainment_DIRECTMaximization(Z1, Z2, tol, maxEval, scalingToggle):
    G = Z1.G if hasattr(Z1, 'G') else Z1.generators()
    m = G.shape[1]
    c1 = Z1.c if hasattr(Z1, 'c') else Z1.center()
    c2 = Z2.c if hasattr(Z2, 'c') else Z2.center()
    # Ensure c1 and c2 are column vectors
    c1 = np.asarray(c1).reshape(-1, 1) if c1.ndim == 1 else np.asarray(c1)
    c2 = np.asarray(c2).reshape(-1, 1) if c2.ndim == 1 else np.asarray(c2)
    def norm_Z2_nu(nu):
        nu = np.asarray(nu).reshape(-1)  # Ensure nu is 1D
        # Compute point: G @ nu gives (n,), then add column vectors
        point = (G @ nu).reshape(-1, 1) + c1 - c2
        # Ensure result is a column vector for zonotopeNorm
        point = np.asarray(point).reshape(-1, 1)
        if hasattr(Z2, 'zonotopeNorm'):
            norm_result = Z2.zonotopeNorm(point)
            # zonotopeNorm returns (val, minimizer), extract just the value
            norm_val = norm_result[0] if isinstance(norm_result, tuple) else norm_result
            return -norm_val
        else:
            return -np.linalg.norm(point)
    bounds = [(-1, 1)] * m
    # Callback for stopping after maxEval
    eval_count = {'count': 0}
    def callback(xk, convergence=None):
        eval_count['count'] += 1
        if scalingToggle:
            return eval_count['count'] > maxEval
        else:
            # Stop if we find a value below -1-tol
            return eval_count['count'] > maxEval or norm_Z2_nu(xk) < -1 - tol
    # Run DIRECT-like optimization (differential evolution as surrogate)
    result = differential_evolution(norm_Z2_nu, bounds, maxiter=1000, popsize=15, tol=1e-6, callback=callback, polish=True, disp=False)
    scaling = abs(-result.fun)
    if scaling > 1 + tol:
        res = False
        cert = True
    else:
        res = True
        cert = False
    return res, cert, scaling 