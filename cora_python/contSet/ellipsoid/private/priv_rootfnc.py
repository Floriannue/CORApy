import numpy as np

def priv_rootfnc(p: float, W1: np.ndarray, q1: np.ndarray, W2: np.ndarray, q2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    priv_rootfnc - polynomial whose root corresponds to the correct p

    Syntax:
        [y,Q,q] = priv_rootfnc(p,W1,q1,W2,q2)

    Inputs:
        p - current p value
        W1,q1,W2,q2 - shape matrices and centers of E1, E2 (see 'and_')

    Outputs:
        y - current function value (should go to 0)
        Q,q - current solution, i.e., ellipsoid(Q,q)

    References: Very heavily inspired by file 'ell_fusionlambda.m' of the
    Ellipsoidal toolbox:
    https://de.mathworks.com/matlabcentral/fileexchange/21936-ellipsoidal-toolbox-et

    Authors:       Victor Gassmann
    Written:       14-October-2019
    Last update:   ---
    Last revision: ---
                   Automatic python translation: Florian Nüssel BA 2025
    """

    n = W1.shape[0]
    X = p * W1 + (1 - p) * W2
    X_inv = np.linalg.inv(X)
    X_inv = 0.5 * (X_inv + X_inv.T)

    a = 1 - p * (1 - p) * (q2 - q1).T @ W2 @ X_inv @ W1 @ (q2 - q1)
    q = X_inv @ (p * W1 @ q1 + (1 - p) * W2 @ q2)
    Q = a * np.linalg.inv(X)
    detX = np.linalg.det(X)

    y = a * detX**2 * np.trace(X_inv @ (W1 - W2)) - n * detX**2 * (
        2 * q.T @ (W1 @ q1 - W2 @ q2)
        + q.T @ (W2 - W1) @ q
        - q1.T @ W1 @ q1
        + q2.T @ W2 @ q2
    )

    return y.item(), Q, q # .item() to convert 0-d array to scalar