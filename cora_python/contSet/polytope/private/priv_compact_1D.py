"""
priv_compact_1D - removes all redundant constraints for a 1D polytope

Description:
    Removes all redundant constraints for a 1D polytope by finding the
    outermost bounds and checking for feasibility.

Syntax:
    A, b, Ae, be, empty = priv_compact_1D(A, b, Ae, be, tol)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    tol - tolerance

Outputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    empty - true/false whether polytope is empty

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from .priv_normalize_constraints import priv_normalize_constraints


def priv_compact_1D(A, b, Ae, be, tol):
    """
    Removes all redundant constraints for a 1D polytope
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        tol: tolerance
        
    Returns:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        empty: true/false whether polytope is empty
    """
    empty = False
    
    if Ae is not None and Ae.size > 0:
        # Normalization yields equality constraints of the form
        #   ax = 0, ax = 1
        # normalize with respect to b
        A, b, Ae, be = priv_normalize_constraints(A, b, Ae, be, 'b')
        
        # More than one unique equality constraint is not feasible
        if Ae.shape[0] > 1:
            if np.any(withinTol(be, 1, tol)) and np.any(withinTol(be, 0, tol)):
                # any combination Ax = 1 and Ax = 0 -> empty
                empty = True
                return np.array([]).reshape(0, 1), np.array([]), np.array([]).reshape(0, 1), np.array([]), empty
            
            # Indices where b = 1 and b = 0
            idx_1 = withinTol(be, 1, tol)
            Ae_1 = Ae[idx_1]
            idx_0 = withinTol(be, 0, tol)
            Ae_0 = Ae[idx_0]
            
            if np.sum(idx_1) > 1 and not np.all(withinTol(Ae_1, Ae_1[0], tol)):
                # More than one constraint with ... = 1
                # and not the same value in A -> empty
                empty = True
                return np.array([]).reshape(0, 1), np.array([]), np.array([]).reshape(0, 1), np.array([]), empty
            elif np.sum(idx_0) > 1 and not np.all(withinTol(Ae_0, Ae_0[0], tol)):
                # More than one constraint with ... = 0
                # and not the same value in A -> empty
                empty = True
                return np.array([]).reshape(0, 1), np.array([]), np.array([]).reshape(0, 1), np.array([]), empty
            
            # Check if we have contradictory equality constraints
            # For the case Ae = [1; 4], be = [2; -5], after normalization we get
            # different values that are not compatible
            if not np.all(withinTol(Ae, Ae[0], tol)):
                # Different coefficients in equality constraints -> check if they're contradictory
                # Normalize to get the actual values: x = be[0]/Ae[0], x = be[1]/Ae[1]
                x1 = be[0] / Ae[0, 0]
                x2 = be[1] / Ae[1, 0]
                if not withinTol(x1, x2, tol):
                    # Contradictory equality constraints -> empty set
                    empty = True
                    return np.array([]).reshape(0, 1), np.array([]), np.array([]).reshape(0, 1), np.array([]), empty
            
            # All equality constraints are the same as the first one; take
            # this one and normalize w.r.t Ae
            Ae = np.array([[1]])
            be = np.array([be[0] / Ae[0, 0]])
    
    if A is not None and A.size > 0:
        # Normalize rows of A matrix
        A, b, _, be_ = priv_normalize_constraints(A, b, Ae, be, 'A')
        
        # Take outermost halfspaces
        idxPos = A.flatten() > 0
        A = np.array([[1], [-1]])
        b_pos = b[idxPos] if np.any(idxPos) else []
        b_neg = b[~idxPos] if np.any(~idxPos) else []
        
        if len(b_pos) == 0:
            A = np.array([[-1]])
            b = np.array([np.min(b_neg)])
        elif len(b_neg) == 0:
            A = np.array([[1]])
            b = np.array([np.min(b_pos)])
        else:
            b = np.array([np.min(b_pos), np.min(b_neg)])
        
        # Check if empty (including equality constraint)
        if len(b) == 2 and b[0] < -b[1]:
            # Constraints of the form ax <= b, ax >= b_, where b_ > b -> infeasible!
            empty = True
            return np.array([]).reshape(0, 1), np.array([]), np.array([]).reshape(0, 1), np.array([]), empty
        elif be is not None and be.size > 0:
            # There are equality constraints
            if len(b) == 2:
                if np.any(be_ > b[0]) or np.any(be_ < -b[1]):
                    # Additionally, an equality constraint that does not comply
                    # with the inequality constraints
                    empty = True
                    return np.array([]).reshape(0, 1), np.array([]), np.array([]).reshape(0, 1), np.array([]), empty
                else:
                    # Equality constraint is satisfied by inequality constraints
                    # BUT: we should keep the inequalities if they define a bounded line segment
                    # Only remove them if the equality constraint completely determines the point
                    if len(b) == 2 and b[0] >= -b[1]:
                        # Keep inequalities as they define a bounded interval
                        pass
                    else:
                        # Only keep equality constraint
                        A = np.array([]).reshape(0, 1)
                        b = np.array([])
            elif len(b) == 1:
                if (A[0, 0] > 0 and np.any(be_ > b[0])) or (A[0, 0] < 0 and np.any(be_ < b[0])):
                    empty = True
                    return np.array([]).reshape(0, 1), np.array([]), np.array([]).reshape(0, 1), np.array([]), empty
                else:
                    # Keep the inequality constraint as it defines a bound
                    pass
    
    return A, b, Ae, be, empty 