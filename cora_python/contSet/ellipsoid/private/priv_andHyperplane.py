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
from cora_python.g.functions.matlab.init.unitvector import unitvector
from cora_python.g.functions.helper.sets.contSet.ellipsoid.vecalign import vecalign

def priv_andHyperplane(E: Ellipsoid, P: Polytope) -> Ellipsoid:
    n = E.dim()
    
    # Store original ellipsoid for potential early return
    E_original = E.copy()

    # Check if the intersection between the ellipsoid and the hyperplane is empty using distance
    # Assuming E.distance(P) is implemented and accurate.
    dist_val = E.distance(P)
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
        Ae_val = np.squeeze(P_.Ae)
        be_val = np.squeeze(P_.be)
        

        
        # Handle empty array case
        if Ae_val.size == 0 or be_val.size == 0:
            # If either Ae or be is empty, this means the hyperplane constraint 
            # became trivial after projection - the ellipsoid is entirely in the hyperplane
            # Return the original ellipsoid since it's entirely contained in the hyperplane
            return E_original
        
        # Convert to scalars if they are size-1 arrays
        if hasattr(Ae_val, 'item') and Ae_val.size == 1:
            Ae_scalar = Ae_val.item()
        else:
            Ae_scalar = Ae_val
        
        if hasattr(be_val, 'item') and be_val.size == 1:
            be_scalar = be_val.item()
        else:
            be_scalar = be_val
        
        # Check for division by zero
        if np.abs(Ae_scalar) < np.finfo(float).eps:
            # If Ae is zero, the hyperplane equation is 0*x = be
            if np.abs(be_scalar) < np.finfo(float).eps:
                # 0*x = 0, hyperplane is the whole space, intersection is the ellipsoid itself
                # Return the original ellipsoid since it's entirely contained in the hyperplane
                return E_original
            else:
                # 0*x = be (be != 0), hyperplane is empty, intersection is empty
                return Ellipsoid.empty(n)
        else:
            # Normal case: Ae is not zero, compute intersection
            xH = be_scalar / Ae_scalar

            # MATLAB: r_xH = max(abs(xH)) * E.TOL;
            # Handle cases where xH might be inf or NaN (shouldn't happen now with the check above)
            xH_scalar = xH
            if np.isinf(xH_scalar) or np.isnan(xH_scalar):
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

        # MATLAB: E = -unitvector_1*P_.be + S*E;
        # This line combines rotation (S*E) and translation (-unitvector_1*P_.be)
        # Apply rotation first, then translation
        E_rotated = S @ E
        # Apply translation: subtract unitvector_1*P_.be from the center
        translation_vec = unitvector_1 * np.squeeze(P_.be)
        E_rotated.q = E_rotated.q - translation_vec.reshape(-1, 1)

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
        
        # The algorithm is correct as implemented - no scaling needed

        Ew = Ellipsoid(W_s,w_s)
        
        # MATLAB: E_t = S'*Ew + P_.be*P_.Ae';
        # This combines back-rotation (S'*Ew) and translation (+ P_.be*P_.Ae')
        E_t = S.T @ Ew # Back-rotate Ew
        
        # Apply the translation: add P_.be*P_.Ae' to the center
        translation_back = np.squeeze(P_.be) * P_.Ae.T
        E_t.q = E_t.q + translation_back.reshape(-1, 1)
    
    # degenerate case: reintroduce x_rem and backtransform
    if is_deg:
        # MATLAB: E = ellipsoid([E_t.Q,zeros(n_nd,n_rem);zeros(n_rem,n)],[E_t.q;x_rem]);
        # This is a bit tricky: it creates a new ellipsoid where Q is diagonal
        # and q is concatenated with x_rem.
        # E.Q is already in transformed space and diagonal, so just need to ensure diagonal elements.
        # What if E.Q is a scalar (1D case after projection)? It should be a 1x1 matrix.
        # Check if empty ellipsoid
        if E_t.Q.size == 0 and E_t.q.size == 0:
            return Ellipsoid.empty(n)

        # Handle the Q_concat properly based on E_t.Q dimensions
        if E_t.Q.ndim == 0: # Scalar case
            Q_concat = np.array([[E_t.Q]])
        elif E_t.Q.size > 0:
            Q_concat = E_t.Q
        else:
            Q_concat = np.zeros((n_nd, n_nd))

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