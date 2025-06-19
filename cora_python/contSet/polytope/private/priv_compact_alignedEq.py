"""
priv_compact_alignedEq - removes all redundant aligned constraints

Description:
    Removes all redundant aligned constraints
    note: expects normalized constraints with respect to 'A'

Syntax:
    Ae, be, empty = priv_compact_alignedEq(Ae, be, tol)

Inputs:
    Ae - equality constraint matrix
    be - equality constraint offset
    tol - tolerance

Outputs:
    Ae - equality constraint matrix
    be - equality constraint offset
    empty - true/false whether polytope is empty

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.auxiliary.withinTol import withinTol


def priv_compact_alignedEq(Ae, be, tol):
    """
    Removes all redundant aligned constraints
    
    Args:
        Ae: equality constraint matrix
        be: equality constraint offset
        tol: tolerance
        
    Returns:
        Ae: equality constraint matrix
        be: equality constraint offset
        empty: true/false whether polytope is empty
    """
    # Assume resulting set is non-empty
    empty = False
    
    if Ae is None or Ae.size == 0:
        return Ae, be, empty
    
    # Number of equality constraints and dimension
    nrConEq, n = Ae.shape
    
    # Try to find aligned constraint vectors -> either infeasible 
    # (different value for be) or redundant (same value for be)
    
    # Pre-compute dot product
    dotprod_norm = Ae @ Ae.T
    
    # Init logical array for which constraints to check
    irredundantEq = np.ones(Ae.shape[0], dtype=bool)
    
    for i in range(nrConEq):
        # If there are aligned constraints, then they have to have the
        # same value in be (all but one of them are redundant),
        # otherwise the polytope is the empty set because there is no
        # point that can satisfy to parallel Ae*x = be at the same time
        
        # Only check if the i-th equality constraint has not already
        # been removed due to alignment
        if irredundantEq[i]:
            # Check for aligned vectors
            alignedConstraints = np.all(
                withinTol(Ae[i, :] - dotprod_norm[:, i].reshape(-1, 1) * Ae[i, :],
                         np.zeros((nrConEq, n)), tol), axis=1)
            
            if np.sum(alignedConstraints) > 1:
                # At least two constraints are aligned
                if np.all(withinTol(be[alignedConstraints], be[i], tol)):
                    # Remove all constraints but the first one
                    irredundantEq[alignedConstraints] = False
                    irredundantEq[i] = True
                else:
                    # Polytope is the empty set
                    empty = True
                    return Ae, be, empty
    
    # Remove all redundant equality constraints
    Ae = Ae[irredundantEq, :]
    be = be[irredundantEq]
    
    return Ae, be, empty 