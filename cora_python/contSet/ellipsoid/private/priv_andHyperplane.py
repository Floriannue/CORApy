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
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
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
    if E.distance(P) > E.TOL:
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
        E = E.transform(U_E.T) # Assuming a transform method or manual application

        # transform hyperplane (possible since T is unitary)
        # MATLAB: P = polytope([],[],(T'*P.Ae')',P.be);
        # Polytope constructor: Polytope(A, b, Aeq, beq)
        # Here, P is a hyperplane, so it has Aeq, beq and no A, b
        # MATLAB: P.Ae is Aeq (matrix), P.be is beq (vector)
        P_Ae_transformed = (U_E.T @ P.Aeq.T).T
        P = Polytope(np.array([[]]).reshape(0, n), np.array([[]]).reshape(0, 1), P_Ae_transformed, P.beq)

        # project ellipsoid and adjust hyperplane
        # MATLAB: x_rem = E.q(n_subspace+1:end);
        x_rem = E.q[n_subspace:]

        # MATLAB: E = project(E,1:n_subspace);
        E = E.project(np.arange(1, n_subspace + 1)) # project uses 1-based indexing for now

        # MATLAB: P = polytope([],[],P.Ae(1:n_subspace),...
        #                    P.be-P.Ae(n_subspace+1:end)*x_rem);
        # Adjust P.Aeq and P.beq for the projected subspace
        P_Ae_proj = P.Aeq[:, :n_subspace]
        P_beq_adj = P.beq - P.Aeq[:, n_subspace:] @ x_rem
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
        xH = P_.beq[0, 0] / P_.Aeq[0, 0] # This will be a float

        # MATLAB: r_xH = max(abs(xH)) * E.TOL;
        r_xH = np.abs(xH) * E.TOL # If xH is a single float, max(abs(xH)) is just abs(xH)
        # Or should it be max(abs(E.q)) * E.TOL if xH is a point on the line?
        # Given MATLAB example: max(abs(xH)) implies xH is the point on the line.

        # MATLAB: IntE_TOL = IntE + interval(-r_xH,r_xH);
        IntE_TOL = IntE + Interval(-r_xH, r_xH)

        # MATLAB: if ~contains_(IntE_TOL,xH,'exact',0,0,false,false)
        # Assuming Interval.contains works with a scalar point
        if not IntE_TOL.contains(xH):
            return Ellipsoid.empty(n)

        # MATLAB: E_t = ellipsoid(0,xH);
        E_t = Ellipsoid(np.zeros((n_nd, n_nd)), np.array([[xH]]))

    else:
        # Higher dimension non-degenerate case
        # compute transformation matrix so that e_1 = S*P.be;

        # for more detail on the following, see [1]
        # MATLAB: unitvector_1 = unitvector(1,n_nd);
        unitvector_1 = unitvector(1, n_nd)

        # MATLAB: S = vecalign(unitvector_1,P_.Ae');
        S = vecalign(unitvector_1, P_.Aeq.T) # P_.Aeq.T because it's a normal vector

        # MATLAB: E = -unitvector_1*P_.be + S*E;
        # Assuming ellipsoid supports scalar multiplication, addition with scalar*vector, and matrix multiplication
        # For scalar * P.be, P.be is (1,1), P_.Aeq.T is (n_nd, 1)
        # This line seems to be transforming the ellipsoid E and centering it w.r.t the hyperplane
        # Reinterpret: E = - (unitvector_1 @ P_.beq) + (S @ E) -> This is wrong for an ellipsoid object
        # This should be a coordinate transformation applied to the ellipsoid
        # MATLAB: S*E implies matrix multiplication with the ellipsoid's Q matrix and center q
        # MATLAB: S*E is (S*E.Q*S', S*E.q)
        # So E_new.Q = S @ E.Q @ S.T
        # E_new.q = S @ E.q - unitvector_1 @ P_.beq

        # Apply the rotation S to the ellipsoid
        E = E.transform(S)
        # Translate the ellipsoid such that the hyperplane defined by P_ passes through the origin
        # The hyperplane equation is P.Aeq @ x = P.beq
        # If P_.Aeq is [a,b] and P_.beq is [c], the distance from origin to hyperplane is c / sqrt(a^2+b^2)
        # If the first coordinate is aligned with the hyperplane normal, then P_.Aeq is something like [1,0,0...]
        # and P_.beq is the signed distance from the origin.
        # E.q = E.q - (unitvector_1 @ P_.beq) seems to be the translation.
        E.q = E.q - unitvector_1 * P_.beq[0,0] # P_.beq is (1,1) matrix, so P_.beq[0,0] is scalar


        # transformed hyperplane (not needed?)
        # P_ = polytope([],[],unitvector_1',0);
        # This line is commented out in MATLAB, indicating it's for reference or debugging.

        # Calculate intermediate matrices for the intersection ellipsoid
        # M = inv(E.Q);
        M = np.linalg.inv(E.Q)

        # Mb = M(2:end,2:end);
        Mb = M[1:, 1:]
        # mb = M(2:end,1);
        mb = M[1:, 0:1] # Column vector
        # m11 = M(1,1);
        m11 = M[0, 0]

        # Mbinv = inv(Mb);
        Mbinv = np.linalg.inv(Mb)

        # w_s = E.q+E.q(1)*[-1;Mbinv*mb];
        # E.q is (n_nd, 1), E.q(1) is first element
        # [-1;Mbinv*mb] needs to be (n_nd, 1)
        # The [-1; Mbinv*mb] should be a column vector.
        temp_vec_part = Mbinv @ mb
        w_s_components = np.vstack([-1.0, temp_vec_part])
        w_s = E.q + E.q[0, 0] * w_s_components


        # a = 1-E.q(1)^2*(m11-mb'*Mbinv*mb);
        a = 1.0 - E.q[0, 0]**2 * (m11 - mb.T @ Mbinv @ mb)
        # Ensure 'a' is a scalar for comparison (mb.T @ Mbinv @ mb is (1,1) matrix)
        if isinstance(a, np.ndarray):
            a = a[0,0]

        if a < 0 and a > -E.TOL:
            a = 0.0
        elif a < -E.TOL:
            raise CORAerror('CORA:specialError',\
                'Error computing intersection of ellipsoid and hyperplane!')

        # W_s = a*[zeros(1,n_nd);[zeros(n_nd-1,1),Mbinv]];
        # This needs to be a symmetric matrix to be part of an ellipsoid Q matrix.
        # The MATLAB construction implies a symmetric matrix. Let's check the dimensions.
        # zeros(1,n_nd) -> (1, n_nd) row of zeros
        # [zeros(n_nd-1,1),Mbinv] -> (n_nd-1, 1) column of zeros, and (n_nd-1, n_nd-1) Mbinv
        # The full matrix is (n_nd, n_nd)
        # W_s structure: [[0, 0, ..., 0],
        #                 [0, Mbinv_11, Mbinv_12, ...],
        #                 [0, Mbinv_21, Mbinv_22, ...],
        #                 ...
        #                 [0, ..., Mbinv_nn]]
        # This forms a (n_nd x n_nd) matrix.
        W_s_block = np.zeros((n_nd, n_nd))
        W_s_block[1:, 1:] = Mbinv # Fills from second row, second column
        W_s = a * W_s_block


        # Ew = ellipsoid(W_s,w_s);
        Ew = Ellipsoid(W_s, w_s)

        # E_t = S'*Ew + P_.be*P_.Ae';
        # S' * Ew implies S.T @ Ew.Q @ S and S.T @ Ew.q
        # P_.be * P_.Ae' implies a translation vector P_.be * P_.Aeq.T
        # This should be back-transforming and translating the ellipsoid
        E_t = Ew.transform(S.T)
        # P_.be is (1,1), P_.Aeq.T is (n_nd, 1)
        E_t.q = E_t.q + P_.beq[0,0] * P_.Aeq.T # Translate by the original hyperplane constant and its normal

    # degenerate case: reintroduce x_rem and backtransform
    if is_deg:
        # MATLAB: E = ellipsoid([E_t.Q,zeros(n_nd,n_rem);zeros(n_rem,n)],[E_t.q;x_rem]);
        # Create the (n x n) Q matrix and (n x 1) q vector for the final ellipsoid
        Q_final = np.zeros((n, n))
        Q_final[:n_nd, :n_nd] = E_t.Q

        q_final = np.vstack([E_t.q, x_rem])

        E_transformed_back = Ellipsoid(Q_final, q_final)

        # MATLAB: E = T*E; (T is U_E from svd of original E.Q)
        # E_transformed_back is in the rotated subspace, now transform back to original coordinates
        E = E_transformed_back.transform(U_E) # Apply original U_E to transform back
    else:
        E = E_t

    return E 