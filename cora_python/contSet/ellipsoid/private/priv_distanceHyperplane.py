import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.polytope.polytope import Polytope

def priv_distanceHyperplane(E: Ellipsoid, P: Polytope) -> float:
    """
    priv_distanceHyperplane - computes the distance from an ellipsoid to a
    hyperplane

    Syntax:
        res = priv_distanceHyperplane(E,P)

    Inputs:
        E - ellipsoid object
        P - polytope object representing a hyperplane

    Outputs:
        val - distance between ellipsoid and hyperplane

    References:
        [1] Kurzhanskiy, A.A. and Varaiya, P., 2006, December. Ellipsoidal toolbox (ET).
            In Proceedings of the 45th IEEE Conference on Decision and Control (pp. 1498-1503). IEEE.
        [2] https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-46.pdf
            (for details)
    """

    # from Kurzhanskiy, A.A. and Varaiya, P., 2006, sec 2.1, eq 2.10
    # can be <0
    # The hyperplane is defined by c'x = y
    if P.Ae.shape[0] != 1:
        raise ValueError("Polytope must represent a single hyperplane (exactly one equality constraint).")
    
    y = P.be[0]
    c = P.Ae.T # c should be a column vector, P.Ae is 1xn, so transpose it

    q = E.q
    Q = E.Q
    
    c_norm_sq = c.T @ c

    # Handle the edge case where the hyperplane's normal vector 'c' is zero.
    # The MATLAB implementation crashes in this case.
    if c_norm_sq < np.finfo(float).eps:
        if np.abs(y) < np.finfo(float).eps:
            # Equation is 0=0, hyperplane is the whole space. Distance is -infinity.
            return -np.inf
        else:
            # Equation is 0=y (y!=0), hyperplane is the empty set. Distance is +infinity.
            return np.inf

    numerator = np.abs(y - c.T @ q) - np.sqrt(c.T @ Q @ c)
    denominator = np.sqrt(c_norm_sq)
    
    val = numerator / denominator

    return float(val) 