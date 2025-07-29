"""
priv_andHyperplane - computes the exact intersection of an ellipsoid and
   a hyperplane

Syntax:
   E = priv_andHyperplane(E,P)

Inputs:
   E - ellipsoid object
   P - polytope object representing a hyperplane

Outputs:
   E - ellipsoid representing the intersection

References:
   [1] A. Kurzhanski et al. "Ellipsoidal Toolbox Manual", 2006
       https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-46.pdf

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ellipsoid/and_

Authors:       Victor Gassmann (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       09-March-2021 (MATLAB)
Last update:   ---
Last revision: ---
"""

import numpy as np
from typing import Union, List, Any

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.isApproxSymmetric import isApproxSymmetric
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.interval.interval import Interval # For 1D case

# Helper functions and methods from other modules
# Assuming these are available as methods on the objects
# E.dim()
# E.distance(P)
# E.isemptyobject() # Already used in or_ method
# E.isFullDim()
# E.rank()
# E.contains(P) for E_is_0D point check
# P.contains(E.q)
# E.project(indices)
# P.normalizeConstraints('A')

# Global helper functions
from cora_python.g.functions.helper.general.unitvector import unitvector
from cora_python.g.functions.helper.sets.contSet.ellipsoid.vecalign import vecalign

def priv_andHyperplane(E: Ellipsoid, P: Polytope) -> Ellipsoid:
    n = E.dim()

    # Check if the intersection between the ellipsoid and the hyperplane is empty using distance
    # Assuming E.distance(P) is implemented and accurate.
    dist_val = E.distance(P)
    print(f"[DEBUG] priv_andHyperplane: dist_val = {dist_val}, E.TOL = {E.TOL}")
    if np.isinf(dist_val) or np.isnan(dist_val) or dist_val > E.TOL:
        return Ellipsoid.empty(n)

    # Check for degeneracy
    is_deg = not E.isFullDim()
    if is_deg:
        n_subspace = E.rank()
        # if E.Q is all zero (0-dimensional ellipsoid/point)
        if n_subspace == 0:
            # if E is 0-d, the result is either E.q if E.q in H, or empty set
            if P.contains(E.q):
                return Ellipsoid(np.zeros((n, n)), E.q)
            else:
                return Ellipsoid.empty(n)

        # If E is degenerate but not 0-dimensional, transform E and P
        # MATLAB: [T,~,~] = svd(E.Q);
        U_E, _, _ = np.linalg.svd(E.Q)
        # transform E such that degenerate dimensions are axis-aligned
        E = U_E.T @ E

        # transform hyperplane (possible since T is unitary)
        # MATLAB: P = polytope([],[],(T'*P.Ae')',P.be);
        # Polytope constructor: Polytope(A, b, Aeq, beq)
        # Here, P is a hyperplane, so it has Aeq, beq and no A, b
        # MATLAB: P.Ae is Aeq (matrix), P.be is beq (vector)
        P_Ae_transformed = P.Ae @ U_E # Corrected: Removed unnecessary .T
        P = Polytope(np.array([[]]).reshape(0, n), np.array([[]]).reshape(0, 1), P_Ae_transformed, P.be)

        # project ellipsoid and adjust hyperplane
        # MATLAB: x_rem = E.q(n_subspace+1:end);
        x_rem = E.q[n_subspace:]

        # MATLAB: E = project(E,1:n_subspace);
        E = E.project(np.arange(n_subspace)) # project uses 0-based indexing

        # MATLAB: P = polytope([],[],P.Ae(1:n_subspace),...
        #                    P.be-P.Ae(n_subspace+1:end)*x_rem);
        # Adjust P.Aeq and P.beq for the projected subspace
        P_Ae_proj = P.Ae[:, :n_subspace]
        P_beq_adj = P.be - P.Ae[:, n_subspace:] @ x_rem
        P = Polytope(np.array([[]]).reshape(0, n_subspace), np.array([[]]).reshape(0, 1), P_Ae_proj, P_beq_adj)

        n_rem = n - n_subspace

    n_nd = E.dim() # Dimension of the non-degenerate ellipsoid

    # normalize hyperplane
    # Assuming normalizeConstraints is a standalone function that takes a Polytope
    # and returns a new Polytope with normalized constraints.
    P_ = P.normalizeConstraints('A') # For hyperplane, A is Aeq

    # Check if only 1 non-degenerate dimension remains in E_nd
    if n_nd == 1:
        # ellipsoid and hyperplane are 1D: check if intervals intersect

        # compute enclosing interval
        # MATLAB: IntE = E.q + interval(-sqrt(E.Q),sqrt(E.Q));
        # For 1D, E.Q is a scalar, so sqrt(E.Q) is also scalar.
        # E.q is a (1,1) array.
        center_1d = E.q[0, 0]
        radius_1d = np.sqrt(E.Q[0, 0]) # E.Q is (1,1) matrix, so access element directly
        IntE = Interval(center_1d - radius_1d, center_1d + radius_1d)

        # MATLAB: xH = P_.be / P_.Ae';
        # For 1D hyperplane, P_.Aeq is (1,1) and P_.beq is (1,1)
        # Aeq is effectively a scalar, beq is a scalar.
        xH = np.squeeze(P_.be) / np.squeeze(P_.Ae) # Use squeeze to ensure scalar for division

        # MATLAB: r_xH = max(abs(xH)) * E.TOL;
        # Handle cases where xH might be inf or NaN from division by zero
        if np.isinf(xH) or np.isnan(xH):
            return Ellipsoid.empty(n_nd)  # If xH is infinite/NaN, intersection is empty

        r_xH = np.abs(xH) * E.TOL

        IntE_TOL = IntE + Interval(-r_xH, r_xH)

        if not IntE_TOL.contains(xH):
            return Ellipsoid.empty(n)

        E_t = Ellipsoid(np.zeros((n_nd, n_nd)), np.array([[xH]]))

    else:
        # Higher dimension non-degenerate case
        # compute transformation matrix so that e_1 = S*P.be;

        # for more detail on the following, see [1]
        # MATLAB: unitvector_1 = unitvector(1,n_nd);
        unitvector_1 = unitvector(1, n_nd)

        # MATLAB: S = vecalign(unitvector_1,P_.Ae');
        S = vecalign(unitvector_1, P_.Ae.T) # P_.Ae.T because it's a normal vector

        # Apply rotation to ellipsoid E
        E_rotated = S @ E # This is E_rotated in MATLAB

        # MATLAB: E = -unitvector_1*P_.be + S*E;
        # The MATLAB line combines rotation and translation on E.
        # We will perform the rotation, then calculate intermediate values,
        # and apply the translation to the center later.

        # Calculate intermediate matrices for the intersection ellipsoid
        # M = inv(E.Q);
        M = np.linalg.inv(E_rotated.Q)

        # Mb = M(2:end,2:end);
        Mb = M[1:, 1:]
        # mb = M(2:end,1);
        mb = M[1:, 0:1] # Column vector
        # m11 = M(1,1);
        m11 = M[0, 0]

        # Mbinv = inv(Mb);
        Mbinv = np.linalg.inv(Mb)

        # w_s = E.q+E.q(1)*[-1;Mbinv*mb];
        # E_rotated.q is (n_nd, 1), E_rotated.q(1) is first element
        # [-1;Mbinv*mb] needs to be (n_nd, 1)
        temp_vec_part = Mbinv @ mb
        w_s_components = np.vstack([-1.0, temp_vec_part])
        w_s = E_rotated.q + E_rotated.q[0, 0] * w_s_components


        # a = 1-E.q(1)^2*(m11-mb'*Mbinv*mb);
        a = 1.0 - E_rotated.q[0, 0]**2 * (m11 - mb.T @ Mbinv @ mb)
        # Ensure 'a' is a scalar for comparison (mb.T @ Mbinv @ mb is (1,1) matrix)
        if isinstance(a, np.ndarray):
            a = a.item() # Get scalar value from 0-d array

        if a < 0 and a > -E.TOL:
            a = 0.0
        elif a < -E.TOL:
            raise CORAerror('CORA:specialError',\
                'Error computing intersection of ellipsoid and hyperplane!')

        # W_s = a*[zeros(1,n_nd);[zeros(n_nd-1,1),Mbinv]];
        # The first row of the block matrix is [0, ..., 0]
        # The rest is [zeros(n_nd-1,1), Mbinv]
        zeros_first_row = np.zeros((1, n_nd))
        zeros_col_vec = np.zeros((n_nd - 1, 1))
        
        # Handle case where Mbinv might be empty for n_nd = 1
        if n_nd == 1:
            W_s = a * np.array([[0.0]]) # For 1D case, W_s is a 1x1 zero matrix scaled by a
        else:
            W_s = a * np.block([[zeros_first_row], [np.hstack((zeros_col_vec, Mbinv))]])

        Ew = Ellipsoid(W_s,w_s)
        E_t = S.T @ Ew # Back-rotate Ew

        # Apply the translation from the hyperplane
        # MATLAB: E_t = S'*Ew + P_.be*P_.Ae';
        E_t.q = E_t.q + np.squeeze(P_.be) * P_.Ae.T
    
    # degenerate case: reintroduce x_rem and backtransform
    if is_deg:
        # MATLAB: E = ellipsoid([E_t.Q,zeros(n_nd,n_rem);zeros(n_rem,n)],[E_t.q;x_rem]);
        # This is a bit tricky: it creates a new ellipsoid where Q is diagonal
        # and q is concatenated with x_rem.
        # E.Q is already in transformed space and diagonal, so just need to ensure diagonal elements.
        # What if E.Q is a scalar (1D case after projection)? It should be a 1x1 matrix.
        if E_t.Q.size == 0 and E_t.q.size == 0: # Check if Ew was an empty ellipsoid
            new_Q = np.zeros((n,n))
            new_q = np.zeros((n,1))
        elif E_t.Q.ndim == 0: # Handle scalar case for 1D, where Ew might be a point ellipsoid
            Q_concat = np.array([[E_t.Q]])
        else:
            # Ensure it's a diagonal matrix in the subspace if it's not already empty
            Q_concat = np.diag(np.diag(E_t.Q)) if E_t.Q.size > 0 else np.zeros((n_nd, n_nd))

        # Construct the combined Q matrix
        if E_t.Q.size == 0 and E_t.q.size == 0: # If it's an empty set, just return empty
            return Ellipsoid.empty(n)

        Q_top = np.hstack((Q_concat, np.zeros((n_nd, n_rem))))
        Q_bottom = np.hstack((np.zeros((n_rem, n_nd)), np.zeros((n_rem, n_rem))))
        new_Q = np.vstack((Q_top, Q_bottom))
        
        # Construct the combined q vector
        new_q = np.vstack((E_t.q, x_rem))

        E = Ellipsoid(new_Q, new_q, TOL=E.TOL)
        E = U_E @ E
    else:
        E = E_t

    return E 