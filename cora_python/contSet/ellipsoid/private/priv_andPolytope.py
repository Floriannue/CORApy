import numpy as np
from typing import Union, List
from ...ellipsoid import Ellipsoid
from ...polytope import Polytope
from ....g.functions.matlab.init.unitvector import unitvector
from ....g.functions.helper.sets.contSet.ellipsoid.vecalign import vecalign
from ....g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_andPolytope(E: Ellipsoid, P: Polytope, mode: str) -> Ellipsoid:
    """
    priv_andPolytope - computes an inner approximation or outer approximation
    of the intersection between an ellipsoid and a polytope

    Syntax:
        E = priv_andPolytope(E,P,mode)

    Inputs:
        E - ellipsoid object
        P - polytope object
        mode - approximation of the intersection
                    'inner': inner approximation of the intersection
                    'outer': outer approximation of the intersection

    Outputs:
        E - ellipsoid approximating the intersection

    References:
        [1] Kurzhanskiy, A.A. and Varaiya, P., 2006, December. Ellipsoidal
            toolbox (ET). In Proceedings of the 45th IEEE Conference o
            Decision and Control (pp. 1498-1503). IEEE.

    Authors:       Victor Gassmann
    Written:       07-June-2022
    Last update:   05-July-2022 (VG, removed input checks; now in parent function)
    Last revision: 23-September-2024 (MW, integrate andHalfspace)
                   Automatic python translation: Florian Nüssel BA 2025
    """

    # Helper for intersection with halfspace
    def aux_andHalfspace(E: Ellipsoid, P_halfspace: Polytope, mode: str) -> Ellipsoid:
        from ...ellipsoid import Ellipsoid
        from ...emptySet import EmptySet
        from ...zonotope import Zonotope # Placeholder for now, may need to import other set types
        from ...contSet import ContSet
        
        # Imports of methods for ellipsoid
        from ...ellipsoid.distance import distance
        from ...ellipsoid.dim import dim
        from ...ellipsoid.contains_ import contains_
        from ...ellipsoid.isFullDim import isFullDim
        from ...ellipsoid.rank import rank
        from ...ellipsoid.project import project
        from ...ellipsoid.supportFunc_ import supportFunc_

        # Placeholder functions (need actual implementation)
        def priv_compIntersectionParam(W1, q1, W2, q2):
            # Placeholder for now, needs to be translated
            raise NotImplementedError("priv_compIntersectionParam not yet translated")

        def priv_rootfnc(p, W1, q1, W2, q2):
            # Placeholder for now, needs to be translated
            raise NotImplementedError("priv_rootfnc not yet translated")
            
        def withinTol(value, target, tolerance):
            return abs(value - target) <= tolerance
            
        # compute distance to corresponding hyperplane
        # Note: In MATLAB, polytope(A,b) can create a halfspace. In Python, Polytope constructor should handle this.
        dist = distance(E, Polytope([], [], P_halfspace.A, P_halfspace.b))
        n = dim(E)

        # touching, completely inside or outside
        if dist >= -E.TOL:
            if P_halfspace.A @ E.q > P_halfspace.b:
                # completely outside or touching
                if withinTol(dist, 0, E.TOL):
                    # touching
                    # point on hyperplane
                    # In MATLAB, P.A' is transpose, P.b is scalar. P.A' * P.b gives a vector.
                    xh = P_halfspace.A.T * P_halfspace.b
                    # direction resulting in touching point has positive
                    # inner product with xh
                    # In MATLAB, sign((xh-E.q)'*P.A')*P.A'
                    # (xh - E.q).T @ P_halfspace.A.T
                    v = np.sign((xh - E.q).T @ P_halfspace.A.T) * P_halfspace.A.T
                    # compute touching point
                    _, x = supportFunc_(E, v, 'upper')
                    E = Ellipsoid(np.zeros((n, n)), x) # Need to confirm ellipsoid constructor for zeros
                else:
                    E = EmptySet(n)
            # else: E completely inside (or touching) -> E = E; (no change)
            return E

        # ...now established that they are intersecting

        T = np.eye(n)
        x_rem = np.array([])

        if not isFullDim(E):
            nt = rank(E)
            # check if E.Q all zero
            if nt == 0:
                # check if E is contained in the polytope
                if contains_(P_halfspace, E.q, 'exact', 0, 0, False, False):
                    E = Ellipsoid(np.zeros((n, n)), E.q)
                else:
                    E = EmptySet(n)
                return E

            # In MATLAB, [T,~,~] = svd(E.Q)
            # For Python, it returns U, S, Vh. We need U for T.
            U, s, Vh = np.linalg.svd(E.Q)
            T = U # MATLAB svd returns U for [U,S,V], so T should be U
            E = E.transform(T.T) # E = T'*E; in MATLAB -> E.transform(T.T) in Python
            # transform inequality constraint (possible since T invertible)
            # P = polytope(P.A*T,P.b); in MATLAB -> Polytope(P.A @ T, P.b)
            P_halfspace = Polytope([], [], P_halfspace.A @ T, P_halfspace.b)
            # project
            x_rem = E.q[nt:]
            E = project(E, list(range(nt))) # 1:nt in MATLAB means 0 to nt-1 in Python
            # P = polytope(P.A(1:nt),P.b-P.A(nt+1:end)*x_rem);
            # P_halfspace.A[0:nt] corresponds to P.A(1:nt)
            # P_halfspace.A[nt:] corresponds to P.A(nt+1:end)
            P_halfspace = Polytope([], [], P_halfspace.A[:nt], P_halfspace.b - P_halfspace.A[nt:] @ x_rem)
            
        n_nd = dim(E)
        # normalize inequality constraint
        # shift E and P such that P.b = 0 and transform such that c=e_1
        A_norm = P_halfspace.A.T / np.linalg.norm(P_halfspace.A)
        b_norm = P_halfspace.b / np.linalg.norm(P_halfspace.A)
        
        # compute transformation matrix so that e_1 = S*c;
        unit_vector_1 = unitvector(1, n_nd)
        S_align = vecalign(unit_vector_1, A_norm) # S is vecalign output
        P_halfspace = Polytope([], [], unit_vector_1.T, 0)
        E = -b_norm * unit_vector_1 + E.transform(S_align) # E = -b*unit_vector_1 + S*E;


        if mode == 'outer':
            r_s, _ = supportFunc_(E, unit_vector_1, 'lower')
            # makes more sense than ET original: define degenerate ellipsoid that
            # covers the transformed ellipsoid "exactly"
            q2 = np.array([1/2*r_s] + [0]*(n_nd-1)).reshape(-1,1) # Column vector
            W2 = np.diag([4/r_s**2] + [0]*(n_nd-1))
            # also, ET original does not work?
            
            W1 = np.linalg.inv(E.Q)
            q1 = E.q

            p = priv_compIntersectionParam(W1, q1, W2, q2)
            _, Q_nd, q_nd = priv_rootfnc(p, W1, q1, W2, q2)
        else: # mode == 'inner'
            # that is "ellipsoidal toolbox original" (not sure why this works)
            # Assuming 'and_' from Ellipsoid class is available
            E_hyp = E.and_op(P_halfspace, 'outer') # Assuming and_ is correctly translated and accessible
            q2 = E_hyp.q - 2 * np.sqrt(np.max(np.linalg.eigvals(E.Q))) * unit_vector_1
            W2 = (unit_vector_1 @ unit_vector_1.T) * 1 / (4 * np.max(np.linalg.eigvals(E.Q)))
            
            W1 = np.linalg.inv(E.Q)
            q1 = E.q

            b1 = (E.q - E_hyp.q).T @ W1 @ (E.q - E_hyp.q)
            _, xb = supportFunc_(E, -unit_vector_1, 'upper')
            b2 = (q2 - xb).T @ W2 @ (q2 - xb)

            b1 = min(1, b1)
            b2 = min(1, b2)
            if b2 >= 1: # assert(b2<1)
                raise CORAerror('CORA:assertion', 'b2 cannot be >= 1 for inner approximation.')
            
            t1 = (1 - b2) / (1 - b1 * b2)
            t2 = (1 - b1) / (1 - b1 * b2)
            
            # (t1*W1+t2*W2)\(t1*W1*q1+t2*W2*q2);
            q_nd = np.linalg.solve((t1 * W1 + t2 * W2), (t1 * W1 @ q1 + t2 * W2 @ q2))
            W = t1 * W1 + t2 * W2
            Q_nd = (1 - t1 * q1.T @ W1 @ q1 - t2 * q2.T @ W2 @ q2 + q_nd.T @ W @ q_nd) * np.linalg.inv(W)

        E_nd = Ellipsoid(Q_nd, q_nd)
        # revert S transform + shift
        Et = S_align.T @ E_nd + dist * A_norm # S_align' * E_nd + dist * A_norm; in MATLAB

        # restore original dimensions and backtransform
        # MATLAB: ellipsoid([Et.Q,zeros(n_nd,n_rem);zeros(n_rem,n)],[Et.q;x_rem]);
        # Python:
        Q_full = np.zeros((n,n))
        Q_full[:n_nd, :n_nd] = Et.Q
        q_full = np.vstack((Et.q, x_rem.reshape(-1,1)))
        
        E_t = Ellipsoid(Q_full, q_full)
        E = T @ E_t # E = T*E_t; in MATLAB -> E = T @ E_t in Python

        return E

    # compute H-rep of P (is computed if not there; expensive!)
    A, b = P.constraints()

    # loop over each inequality constraint
    for i in range(A.shape[0]):
        # In MATLAB, polytope(A(i,:),b(i)) means a single half-space.
        # Ensure that Polytope constructor can handle this.
        E = aux_andHalfspace(E, Polytope([], [], A[i,:].reshape(1,-1), b[i]), mode)

    return E