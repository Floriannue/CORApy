"""
priv_zonotopeContainment_zonoSampling - Solves the zonotope containment problem by using the
   Shenmaier vertex sampling algorithm described in [1].

Syntax:
   res, cert, scaling = priv_zonotopeContainment_zonoSampling(Z1, Z2, tol, N, scalingToggle)

Inputs:
   Z1 - zonotope object, inbody
   Z2 - zonotope object, circumbody
   tol - tolerance for the containment check
   N - Number of random samples
   scalingToggle - if set to True, scaling will be computed

Outputs:
   res - True/False
   cert - see logic below
   scaling - the smallest number 'scaling', such that scaling*(Z2 - center(Z2)) + center(Z2) contains Z1 (lower bound)

References:
   [1] Kulmburg A., Brkan I., Althoff M.,: Search-based and Stochastic Solutions to the Zonotope and Ellipsotope Containment Problems (to appear)

Authors:       Adrian Kulmburg
Python port:   AI Assistant
"""
import numpy as np

def priv_zonotopeContainment_zonoSampling(Z1, Z2, tol, N, scalingToggle):
    G = Z1.G if hasattr(Z1, 'G') else Z1.generators()
    c = Z1.c if hasattr(Z1, 'c') else Z1.center()
    alphas = np.sign(np.random.rand(G.shape[1], N) - 0.5)
    if scalingToggle:
        from ..contains_ import contains_
        res_arr, _, scaling_arr = contains_(Z2, G @ alphas + c.reshape(-1, 1), 'exact', tol, N, True, True)
        res = np.all(res_arr)
        cert = not res
        scaling = np.max(scaling_arr)
        return res, cert, scaling
    else:
        scaling = np.nan
        res = True
        from ..contains_ import contains_
        for i in range(N):
            p = G @ alphas[:, i] + c
            res_i, _, _ = contains_(Z2, p, 'exact', tol, N, False, False)
            if not res_i:
                res = False
                break
        cert = not res
        return res, cert, scaling 