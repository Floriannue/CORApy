"""
priv_zonotopeContainment_zonoSamplingDual - Solves the zonotope containment problem by using
   the Shenmaier halfspace sampling algorithm described in [1].

Syntax:
   res, cert, scaling = priv_zonotopeContainment_zonoSamplingDual(Z1, Z2, tol, N, scalingToggle)

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

def ndimCross(Q):
    # Compute the normal vector to the hyperplane spanned by columns of Q
    # Q: n x (n-1) matrix
    # Returns: n x 1 vector
    if Q.shape[1] == 1:
        # 2D case
        return np.array([-Q[1, 0], Q[0, 0]])
    elif Q.shape[1] == 2:
        # 3D case
        return np.cross(Q[:, 0], Q[:, 1])
    else:
        # General nD case: use SVD to find null space
        u, s, vh = np.linalg.svd(Q.T)
        null_mask = (s <= 1e-12)
        if np.any(null_mask):
            return vh[-1, :]
        else:
            # Fallback: return last right singular vector
            return vh[-1, :]

def priv_zonotopeContainment_zonoSamplingDual(Z1, Z2, tol, N, scalingToggle):
    c = Z1.c if hasattr(Z1, 'c') else Z1.center()
    G = Z1.G if hasattr(Z1, 'G') else Z1.generators()
    d = Z2.c if hasattr(Z2, 'c') else Z2.center()
    H = Z2.G if hasattr(Z2, 'G') else Z2.generators()
    n = H.shape[0]
    ell = H.shape[1]
    G_prime = np.hstack([G, (c - d).reshape(-1, 1)])
    scaling = 0.0
    for i in range(N):
        permutation = np.random.permutation(ell)
        ind = permutation[:n-1]
        Q = H[:, ind]
        lambda_vec = ndimCross(Q)
        b = np.sum(np.abs(lambda_vec @ H))
        if b == 0:
            lambda_vec = 0 * lambda_vec
        else:
            lambda_vec = lambda_vec / b
        s = np.sum(np.abs(G_prime.T @ lambda_vec))
        scaling = max(scaling, s)
        if not scalingToggle and scaling > 1 + tol:
            break
    res = scaling <= 1 + tol
    cert = not res
    return res, cert, scaling 