import numpy as np
from scipy.sparse import csc_matrix, eye as speye, kron, vstack, hstack
from scipy.optimize import linprog

def priv_zonotopeContainment_SadraddiniTedrake(Z1, Z2, tol, scalingToggle):
    """
    Solves the zonotope containment problem using the method from Sadraddini et al.
    
    This function checks if zonotope Z1 is contained in zonotope Z2 by solving
    a linear program.
    
    Args:
        Z1: The inner zonotope object (inbody).
        Z2: The outer zonotope object (circumbody).
        tol: Tolerance for the containment check.
        scalingToggle: If True, compute and return the scaling factor.
        
    Returns:
        tuple: (res, cert, scaling)
    """
    # Extract generators and centers
    G_inbody = Z1.G if Z1.G is not None else np.empty((Z1.dim, 0))
    G_circum = Z2.G if Z2.G is not None else np.empty((Z2.dim, 0))
    
    c_inbody = Z1.c
    c_circum = Z2.c
    
    # Add the difference of the centers to the generators of the inbody
    center_diff = c_inbody - c_circum
    G_inbody = np.hstack([G_inbody, center_diff])
    
    # Get dimensions
    n = G_inbody.shape[0]
    m_inbody = G_inbody.shape[1]
    m_circum = G_circum.shape[1]
    
    # If circumbody has no generators, containment is only possible if inbody is just a point
    # at the center of the circumbody.
    if m_circum == 0:
        res = np.linalg.norm(G_inbody) < tol
        cert = True
        scaling = 0 if res else np.inf
        return res, cert, scaling

    # Sparsifying matrices
    G_inbody_s = csc_matrix(G_inbody)
    G_circum_s = csc_matrix(G_circum)

    # --- Setting up constraints for the linear program ---
    m_prod = m_inbody * m_circum
    I_prod = speye(m_prod, format='csc')
    
    # A_ub * x <= b_ub
    A1 = hstack([I_prod, -I_prod, csc_matrix((m_prod, 1))])
    A2 = hstack([-I_prod, -I_prod, csc_matrix((m_prod, 1))])
    A3 = hstack([
        csc_matrix((m_circum, m_prod)),
        kron(np.ones((1, m_inbody)), speye(m_circum, format='csc')),
        -np.ones((m_circum, 1))
    ])
    A_ub = vstack([A1, A2, A3], format='csc')
    b_ub = np.zeros(A_ub.shape[0])

    # A_eq * x = b_eq
    A_eq = hstack([
        kron(speye(m_inbody, format='csc'), G_circum_s),
        csc_matrix((n * m_inbody, m_prod)),
        csc_matrix((n * m_inbody, 1))
    ], format='csc')
    b_eq = G_inbody_s.toarray().flatten('F')

    # Objective function: minimize w
    cost = np.zeros(A_ub.shape[1])
    cost[-1] = 1
    
    # Define bounds for the variables x = [Gamma; GammaAux; w]
    gamma_bounds = (None, None) # Unbounded
    gamma_aux_bounds = (0, None)   # Non-negative
    w_bounds = (0, None)           # Non-negative scaling factor

    bounds = [gamma_bounds] * m_prod + [gamma_aux_bounds] * m_prod + [w_bounds]

    # Solve the linear program
    res_lp = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res_lp.success:
        return False, False, np.inf

    scaling_val = res_lp.fun
    res = scaling_val <= 1 + tol
    
    if scalingToggle:
        if res:
            cert = True
        elif scaling_val > np.sqrt(m_inbody):
            cert = True
        else:
            cert = False
        return res, cert, scaling_val
    else:
        cert = res
        return res, cert, np.nan
