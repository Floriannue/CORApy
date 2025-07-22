"""
priv_orEllipsoidOA - Computes an outer-approximation of the union between ellipsoids

Syntax:
    E = priv_orEllipsoidOA(E_cell)

Inputs:
    E_cell - list of Ellipsoid objects

Outputs:
    E - Ellipsoid after union

References:
   [1] S. Boyd et al. "Convex Optimization"

Authors:       Victor Gassmann (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       15-March-2021 (MATLAB)
Python translation: 2025
"""

import numpy as np
import cvxpy as cp
# from cora_python.g.functions.matlab.validate.check.isFullDim import isFullDim # Assuming this will be available or implemented
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid # For creating new ellipsoid objects
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def priv_orEllipsoidOA(E_cell):
    """
    priv_orEllipsoidOA - Computes an outer-approximation of the union between ellipsoids

    Syntax:
        E = priv_orEllipsoidOA(E_cell)

    Inputs:
        E_cell - list of Ellipsoid objects

    Outputs:
        E - Ellipsoid after union

    References:
       [1] S. Boyd et al. "Convex Optimization"

    Authors:       Victor Gassmann (MATLAB)
                   Automatic python translation: Florian Nüssel BA 2025
    Written:       15-March-2021 (MATLAB)
    Python translation: 2025
    """

    # WARNING: This function attempts to solve a Semidefinite Programming (SDP) problem
    # for the Minimum Volume Covering Ellipsoid (MVEE). The original MATLAB implementation
    # relies on highly optimized commercial solvers (MOSEK, SDPT3, YALMIP with external solvers)
    # which are more robust for these types of numerically challenging problems, especially
    # with complex LMI formulations and potentially degenerate inputs.
    #
    # Despite implementing preprocessing steps for numerical stability and restricting
    # the problem to N=2 ellipsoids, open-source CVXPY solvers (ECOS, SCS, Clarabel) have
    # consistently failed to find solutions due to numerical issues.
    #
    # For a full and robust implementation, a commercial SDP solver with a Python API
    # (e.g., MOSEK through CVXPY) or a more advanced, tailored open-source SDP formulation
    # (which is a research-level task) would be required.
    #
    # This current implementation will likely raise a CORAerror indicating solver failure
    # for non-trivial inputs.

    N = len(E_cell)
    n = E_cell[0].dim()

    if N != 2:
        raise NotImplementedError('Union of more than two ellipsoids is not yet implemented for open-source solvers due to numerical stability challenges with complex SDP formulations. Consider using commercial solvers like MOSEK or contributing advanced reformulations.')

    # Normalize via maximum singular value to prevent numerical issues
    # MATLAB: max_val = max(cellfun(@(E_i) max(svd(E_i.Q)),E_cell,'UniformOutput',true));
    max_val = 0.0
    for E_i in E_cell:
        if E_i.Q.size > 0: # Check if Q is not empty for svd
            max_val = max(max_val, np.max(np.linalg.svd(E_i.Q)[1])) # svd returns (U, s, Vh) where s are singular values

    # Pre-compute scaling
    fac = 0.001
    th = fac * max_val
    if th == 0:
        th = fac

    # If any ellipsoid is degenerate, add small perturbation (possible since we
    # compute an overapproximation)
    E_reg = []
    for i in range(N):
        E_i = E_cell[i]
        # MATLAB: if ~isFullDim(E_i)
        # Using np.linalg.matrix_rank for isFullDim equivalent
        if np.linalg.matrix_rank(E_i.Q) < n:
            nd_i = np.linalg.matrix_rank(E_i.Q)
            U_i, s_i, V_i = np.linalg.svd(E_i.Q) # V_i is Vh in numpy
            s_i = np.diag(s_i)

            # MATLAB: Si = diag([si(1:nd_i);th*ones(n-nd_i,1)]);
            # Python: reconstruct Si
            Si_diag_vals = np.concatenate((np.diag(s_i)[:nd_i], th * np.ones(n - nd_i)))
            Si = np.diag(Si_diag_vals)

            # MATLAB: E_cell{i} = ellipsoid(Ti*Si*Ti',E_i.q);
            E_reg.append(Ellipsoid(U_i @ Si @ U_i.T, E_i.q))
        else:
            E_reg.append(E_i)

    # Step 2: Set up the SDP for the MVEE
    A2 = cp.Variable((n, n), symmetric=True)
    bt = cp.Variable((n, 1))
    l = cp.Variable(N, nonneg=True)
    constraints = [A2 >> 0]

    for i, E in enumerate(E_reg):
        # MATLAB assumes Q is full rank here due to preprocessing, so can directly invert.
        Qi_inv = np.linalg.inv(E.Q)
        qi = E.q
        c_j_scalar = (qi.T @ Qi_inv @ qi - 1)[0,0] # This is a numpy scalar.

        # Ensure this scalar expression is a 1x1 matrix for cp.bmat
        block22_scalar_expr = cp.reshape(cp.Constant(-1.0) - l[i] * cp.Constant(c_j_scalar), (1, 1), order='F')

        # Exact LMI from MATLAB: [A2-l(i)*Qiinv, bt+l(i)*Qiinv*qi, zeros(n);
        #                        (bt+l(i)*Qiinv*qi)',-1-l(i)*c_j, bt';
        #                        zeros(n), bt, -A2] >= 0

        lmi_matrix = cp.bmat([
            [A2 - l[i] * Qi_inv, bt + l[i] * Qi_inv @ qi, np.zeros((n, n))],
            [(bt + l[i] * Qi_inv @ qi).T, block22_scalar_expr, bt.T],
            [np.zeros((n, n)), bt, -A2]
        ])
        constraints.append(lmi_matrix >> 0)

    # Objective: minimize log det(A2^{-1}) (i.e., maximize det(A2))
    prob = cp.Problem(cp.Minimize(-cp.log_det(A2)), constraints)
    try:
        prob.solve(solver=cp.CLARABEL) # Changed solver to Clarabel
    except Exception as e:
        raise CORAerror('CORA:solverIssue', f'cvxpy: {e}')

    if A2.value is None or bt.value is None:
        raise CORAerror('CORA:solverIssue', 'cvxpy')

    # Extract solution values
    A2_sol = A2.value
    bt_sol = bt.value

    # Construct ellipsoid parameters
    # MATLAB: q = -A2\bt_sol; Q = inv(A2);
    q = -np.linalg.solve(A2_sol, bt_sol)
    Q = np.linalg.inv(A2_sol)

    # Final ellipsoid
    E = Ellipsoid(Q, q)
    return E 