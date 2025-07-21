"""
priv_zonotopeContainment_ellipsoidSampling - Solves the
   ellipsoid-in-zonotope containment problem by using the Shenmaier vertex
   sampling algorithm described in [1].

Syntax:
   res, cert, scaling = priv_zonotopeContainment_ellipsoidSampling(E, Z, tol, N, scalingToggle)

Inputs:
   E - ellipsoid object, inbody
   Z - zonotope object, circumbody
   tol - tolerance for the containment check
   N - Number of random samples
   scalingToggle - if set to True, scaling will be computed

Outputs:
   res - True/False
   cert - see logic below
   scaling - the smallest number 'scaling', such that scaling*(Z - center(Z)) + center(Z) contains E (lower bound)

References:
   [1] Kulmburg A., Brkan I., Althoff M.,: Search-based and Stochastic Solutions to the Zonotope and Ellipsotope Containment Problems (to appear)

Authors:       Adrian Kulmburg
Python port:   AI Assistant
"""
import numpy as np

def priv_zonotopeContainment_ellipsoidSampling(E, Z, tol, N, scalingToggle):
    Q = E.Q if hasattr(E, 'Q') else E.get_Q()
    q = E.q if hasattr(E, 'q') else E.get_q()
    U, Sigma, _ = np.linalg.svd(Q)
    s = np.diag(Sigma) if Sigma.ndim == 2 else Sigma
    s_plus = 1.0 / s
    s_plus = np.sqrt(s_plus)
    s_plus[np.abs(s) < 1000 * np.finfo(float).eps] = 0
    G = U @ np.diag(s_plus)
    c = q
    d = Z.G.shape[0] if hasattr(Z, 'G') else Z.generators().shape[0]
    p = np.random.randn(d, N)
    p = p / np.sqrt(np.sum(p ** 2, axis=0, keepdims=True))
    p = G @ p + c.reshape(-1, 1)
    from ..contains_ import contains_
    if scalingToggle:
        res_arr, _, scaling_arr = contains_(Z, p, 'exact', tol, N, True, True)
        res = np.all(res_arr)
        cert = not res
        scaling = np.max(scaling_arr)
    else:
        scaling = np.nan
        res = True
        for i in range(N):
            res_i, _, _ = contains_(Z, p[:, i], 'exact', tol, N, False, False)
            if not res_i:
                res = False
                break
        cert = not res
    return res, cert, scaling 