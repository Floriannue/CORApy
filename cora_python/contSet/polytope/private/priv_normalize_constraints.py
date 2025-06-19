import numpy as np
from cora_python.g.functions.matlab.validate.check import withinTol

def priv_normalize_constraints(A, b, Ae, be, type):
    # priv_normalizeConstraints - normalizes the constraints
    #
    # Syntax:
    #    [A,b,Ae,be] = priv_normalizeConstraints(A,b,Ae,be,type)
    #
    # Inputs:
    #    A - inequality constraint matrix
    #    b - inequality constraint offset
    #    Ae - equality constraint matrix
    #    be - equality constraint offset
    #    type - (optional) 'b', 'be': normalize offset vectors b and be
    #                      'A', 'Ae': normalize norm of constraints in A and Ae to 1
    #
    # Outputs:
    #    A - inequality constraint matrix
    #    b - inequality constraint offset
    #    Ae - equality constraint matrix
    #    be - equality constraint offset

    if type == 'b' or type == 'be':
        # normalize offset vectors b and be to -1|0|1

        # normalize inequality constraints
        if A is not None and A.size > 0:
            # inequality constraints where A[:,i]*x <= 0 are left unchanged
        
            # find indices of constraints with b > 0 and b < 0
            idx_plus = b > 0
            idx_neg = b < 0
        
            # divide constraints with b > 0 by b
            if np.any(idx_plus):
                A[idx_plus,:] = A[idx_plus,:] / b[idx_plus, np.newaxis]
                b[idx_plus] = 1
        
            # divide constraints with b < 0 by b, but set b to -1
            if np.any(idx_neg):
                A[idx_neg,:] = A[idx_neg,:] / b[idx_neg, np.newaxis]
                b[idx_neg] = -1

        # normalize equality constraints
        if Ae is not None and Ae.size > 0:
            # equality constraints where Ae[:,i]*x = 0 are left unchanged
        
            # find indices of constraints with be > 0 and be < 0
            idx_nonzero = be != 0
        
            # divide constraints with be =!= 0 by be
            if np.any(idx_nonzero):
                Ae[idx_nonzero,:] = Ae[idx_nonzero,:] / be[idx_nonzero, np.newaxis]
                be[idx_nonzero] = 1

    elif type == 'A' or type == 'Ae':
        # normalize norm of constraints in A and Ae to 1
        # skip constraints of the form 0*x <= ... or 0*x == ...

        # normalize inequality constraints
        if A is not None and A.size > 0:
            normA = np.linalg.norm(A, axis=1)
            idx_nonzero = ~withinTol(normA, 0)
            if np.any(idx_nonzero):
                A[idx_nonzero,:] = A[idx_nonzero,:] / normA[idx_nonzero, np.newaxis]
                b[idx_nonzero] = b[idx_nonzero] / normA[idx_nonzero]

        # normalize equality constraints
        if Ae is not None and Ae.size > 0:
            normA = np.linalg.norm(Ae, axis=1)
            idx_nonzero = ~withinTol(normA, 0)
            if np.any(idx_nonzero):
                Ae[idx_nonzero,:] = Ae[idx_nonzero,:] / normA[idx_nonzero, np.newaxis]
                be[idx_nonzero] = be[idx_nonzero] / normA[idx_nonzero]

    return A, b, Ae, be 