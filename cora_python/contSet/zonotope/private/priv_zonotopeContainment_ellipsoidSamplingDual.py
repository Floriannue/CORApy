"""
priv_zonotopeContainment_ellipsoidSamplingDual - Solves the zonotope containment problem by
   using the Shenmaier halfspace sampling algorithm described in [1].

Syntax:
   res, cert, scaling = priv_zonotopeContainment_ellipsoidSamplingDual(E, Z, tol, N, scalingToggle)

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

def priv_zonotopeContainment_ellipsoidSamplingDual(E, Z, tol, N, scalingToggle):
    Q = E.Q if hasattr(E, 'Q') else E.get_Q()
    q = E.q if hasattr(E, 'q') else E.get_q()
    U, Sigma, _ = np.linalg.svd(Q)
    s = np.diag(Sigma) if Sigma.ndim == 2 else Sigma
    s_plus = 1.0 / s
    s_plus = np.sqrt(s_plus)
    s_plus[np.abs(s) < 1000 * np.finfo(float).eps] = 0
    G = U @ np.diag(s_plus)
    c = q
    H = Z.G if hasattr(Z, 'G') else Z.generators()
    n = H.shape[0]
    ell = H.shape[1]
    d = Z.c if hasattr(Z, 'c') else Z.center()
    G_prime = np.hstack([G, (c - d).reshape(-1, 1)])
    scaling = 0.0
    for i in range(N):
        permutation = np.random.permutation(ell)
        ind = permutation[:n-1]
        Qmat = H[:, ind]
        lambda_vec = ndimCross(Qmat)
        b = np.sum(np.abs(lambda_vec @ H))
        if b == 0:
            lambda_vec = 0 * lambda_vec
        else:
            lambda_vec = lambda_vec / b
        s_val = np.sqrt(np.sum((G_prime.T @ lambda_vec) ** 2))
        scaling = max(scaling, s_val)
        if not scalingToggle and scaling > 1 + tol:
            break
    res = scaling <= 1 + tol
    cert = not res
    return res, cert, scaling 