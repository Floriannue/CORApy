"""
priv_orEllipsoidOA - Computes an outer-approximation of the union between
ellipsoids

Syntax:
   E = priv_orEllipsoidOA(E)

Inputs:
   E_cell - cell-array of ellipsoid objects

Outputs:
   E - ellipsoid after union

References:
  [1] S. Boyd et al. "Convex Optimization"

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ellipsoid/or

Authors:       Victor Gassmann
Written:       15-March-2021
Last update:   05-July-2022 (VG, remove unecessary input)
Last revision: ---

------------------------------ BEGIN CODE -------------------------------
"""

import numpy as np
import cvxpy as cp
import scipy.linalg
from typing import List, TYPE_CHECKING

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid # Added direct import

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

def priv_orEllipsoidOA(E_cell: List['Ellipsoid']) -> 'Ellipsoid':
    # collapse cell array
    N = len(E_cell)
    n = E_cell[0].dim()

    # normalize via maximum singular value to prevent numerical issues
    max_val = 0.0
    for E_i in E_cell:
        if E_i.Q.size > 0:
            current_max_sv = np.max(np.linalg.svd(E_i.Q, compute_uv=False))
            if current_max_sv > max_val:
                max_val = current_max_sv
    
    # pre-compute scaling
    fac = 0.001 
    th = fac * max_val
    if th == 0:
        th = fac # Fallback if max_val was 0 (e.g., all zero matrices)

    # if any ellipsoid is degenerate, add small perturbation (possible since we
    # compute an overapproximation)
    for i in range(len(E_cell)):
        E_i = E_cell[i]
        if not E_i.isFullDim():
            nd_i = E_i.rank()
            Ti, Si_diag, Vh = np.linalg.svd(E_i.Q)
            si = Si_diag # Si_diag is already a 1D array of singular values

            # bloat to remove degeneracy
            Si_bloated = np.diag(np.concatenate((si[:nd_i], th * np.ones(n - nd_i))))
            E_cell[i] = Ellipsoid(Ti @ Si_bloated @ Ti.T, E_i.q)
    
    # CVXPY implementation
    # Find minimum volume ellipsoid spanning union [1]
    
    # Define CVXPY variables
    A2 = cp.Variable((n, n), symmetric=True)
    bt = cp.Variable((n, 1))
    l = cp.Variable(N, pos=True) # l must be positive (>=0)

    constraints = [A2 >> 0] # A2 must be positive semi-definite (PSD) for log_det

    for i in range(N):
        Q_i = E_cell[i].Q
        q_i = E_cell[i].q

        # Handle singular Q_i (due to initial degeneracy or numerical issues)
        try:
            Qinv_i = np.linalg.inv(Q_i)
        except np.linalg.LinAlgError:
            # If Q_i is singular, it means the ellipsoid is degenerate.
            # It should have been perturbed above by `th` so this case should be rare.
            # If it still occurs, it means the perturbation was not enough or it's
            # an edge case, and we might need to handle it more robustly or
            # throw an error if the problem becomes unsolvable.
            # For now, let's assume the perturbation handles it.
            # A more robust solution might involve pseudoinverse or other techniques.
            raise CORAerror('CORA:solverIssue', 'Numerical issue with singular Q matrix after perturbation.')

        c_i = q_i.T @ Qinv_i @ q_i - 1

        # Construct the LMI (Linear Matrix Inequality)
        # The structure is (n+1+n) x (n+1+n)
        # Ci = [A2-l(i)*Qiinv,    bt+l(i)*Qiinv*qi,            zeros(n);
        #       (bt+l(i)*Qiinv*qi)',-1-l(i)*(qi'*Qiinv*qi-1),  bt';
        #       zeros(n),          bt,                         -A2];
        # We need Ci <= 0, which means -Ci >= 0.

        M11 = -A2 + l[i] * Qinv_i
        M12 = -bt - l[i] * Qinv_i @ q_i
        M13 = cp.Constant(np.zeros((n, n)))

        M21 = (-bt - l[i] * Qinv_i @ q_i).T
        M22 = 1 + l[i] * c_i
        M23 = -bt.T

        M31 = cp.Constant(np.zeros((n, n)))
        M32 = -bt
        M33 = A2 
        
        M22_matrix = cp.reshape(M22, (1, 1), order='C') # Added order='C'

        lmi_matrix = cp.bmat([
            [M11, M12, M13],
            [M21, M22_matrix, M23],
            [M31, M32, M33]
        ])
        constraints.append(lmi_matrix >> 0)

    objective = cp.Minimize(-cp.log_det(A2))
    prob = cp.Problem(objective, constraints)

    # Solve the SDP
    try:
        # Try OSQP first for potentially better precision
        prob.solve(solver=cp.OSQP, verbose=True) # Set verbose=True for OSQP
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise cp.SolverError(f"OSQP did not find an optimal solution. Status: {prob.status}")
    except cp.SolverError as e_osqp:
        try:
            # Fallback to CLARABEL
            prob.solve(solver=cp.CLARABEL, verbose=False) 
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise cp.SolverError(f"CLARABEL did not find an optimal solution. Status: {prob.status}")
        except cp.SolverError as e_clarabel:
            try:
                # Fallback to MOSEK (if installed) with verbose output
                prob.solve(solver=cp.MOSEK, verbose=True)
                if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    raise cp.SolverError(f"MOSEK did not find an optimal solution. Status: {prob.status}")
            except cp.SolverError as e_mosek:
                try:
                    # Fallback to SCS with increased accuracy settings
                    prob.solve(solver=cp.SCS, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
                    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        raise cp.SolverError(f"SCS did not find an optimal solution. Status: {prob.status}")
                except cp.SolverError as e_scs:
                    try:
                        # Fallback to ECOS with increased accuracy settings
                        prob.solve(solver=cp.ECOS, verbose=False, eps_abs=1e-6)
                        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                            raise cp.SolverError(f"ECOS did not find an optimal solution. Status: {prob.status}")
                    except cp.SolverError as e_ecos:
                        raise CORAerror(f"All solvers failed for priv_orEllipsoidOA: OSQP ({e_osqp}), CLARABEL ({e_clarabel}), MOSEK ({e_mosek}), SCS ({e_scs}), ECOS ({e_ecos})")
                    except Exception as e_ecos_general:
                        raise CORAerror(f"ECOS solver failed or is not configured: {e_ecos_general}. OSQP failed: {e_osqp}, CLARABEL failed: {e_clarabel}, MOSEK failed: {e_mosek}, SCS failed: {e_scs}")
                except Exception as e_scs_general:
                    raise CORAerror(f"SCS solver failed or is not configured: {e_scs_general}. OSQP failed: {e_osqp}, CLARABEL failed: {e_clarabel}, MOSEK failed: {e_mosek}")
            except Exception as e_mosek_general:
                raise CORAerror(f"MOSEK solver failed or is not configured: {e_mosek_general}. OSQP failed: {e_osqp}, CLARABEL failed: {e_clarabel}")
        except Exception as e_clarabel_general:
            raise CORAerror(f"CLARABEL solver failed or is not configured: {e_clarabel_general}. OSQP failed: {e_osqp}")
    except Exception as e_osqp_general:
        raise CORAerror(f"OSQP solver failed or is not configured: {e_osqp_general}")


    # Extract results
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        A2_sol = A2.value
        bt_sol = bt.value

        # Construct ellipsoid parameters from solver output
        # In MATLAB: Q = inv(A2); q = -A2\bt_sol;
        # In Python: use inverse and solve for q
        Q_sol = np.linalg.inv(A2_sol)
        q_sol = -np.linalg.solve(A2_sol, bt_sol)
        E = Ellipsoid(Q_sol, q_sol)
    else:
        # If solver did not find an optimal or inaccurate solution
        raise CORAerror('CORA:solverIssue', f'SDP solver failed to find an optimal solution. Status: {prob.status}')

    return E 