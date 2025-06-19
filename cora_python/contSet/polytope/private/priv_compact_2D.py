"""
priv_compact_2D - special computation for 2D case without linear programs

Description:
    Special computation for 2D case without linear programs: First, we order 
    the (already normalized) constraints by angle. Then, we check whether the 
    middle constraint between two adjacent constraints is required by computing 
    the intersecting vertex between the outer two constraints and checking 
    whether that middle constraint has a smaller support function value than 
    the intersecting vertex in the direction of the middle constraint.

Syntax:
    A, b, Ae, be, empty = priv_compact_2D(A, b, Ae, be, tol)

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
from .priv_compact_alignedEq import priv_compact_alignedEq


def priv_compact_2D(A, b, Ae, be, tol):
    """
    Removes all redundant constraints for a 2D polytope using specialized 2D algorithm
    
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
        # Remove duplicates in equality constraints
        Ae, be, empty = priv_compact_alignedEq(Ae, be, tol)
        # Emptiness already determined
        if empty:
            return np.array([]), np.array([]), np.array([]), np.array([]), empty
    
    if A is not None and A.size > 0:
        # Transpose constraints, read out number
        A = A.T
        nrCon = len(b)
        
        # Constraints are normalized, order by angle
        angles = np.degrees(np.arctan2(A[1, :], A[0, :]))
        idx = np.argsort(angles)
        angles = angles[idx]
        A = A[:, idx]
        b = b[idx]
        
        # Remove parallel constraints (need to be in order!)
        idxKept = np.ones(nrCon, dtype=bool)
        startIdx = nrCon - 1
        endIdx = 0
        idxUnchecked = np.ones(nrCon, dtype=bool)
        
        # Loop until all have been compared once
        while np.any(idxUnchecked):
            # Compute dot product, check for alignment
            if idxUnchecked[startIdx] and withinTol(A[:, startIdx].T @ A[:, endIdx], 1):
                # Compute dot product with all constraints, compare all which
                # are parallel and keep the one with the smallest offset value
                dotprod = A[:, startIdx].T @ A
                idxParallel = withinTol(dotprod, 1)
                minIdx = np.argmin(b[idxParallel])
                
                # Remove all with larger offset
                idxKept = idxKept & ~idxParallel
                
                # Put back the one with smallest offset value
                parallel_indices = np.where(idxParallel)[0]
                if len(parallel_indices) > minIdx:
                    idxKept[parallel_indices[minIdx]] = True
                
                # All from this batch have been checked -> do not check again
                idxUnchecked[idxParallel] = False
            
            # Increment indices
            idxUnchecked[startIdx] = False
            startIdx = endIdx
            endIdx = endIdx + 1
            if endIdx >= nrCon:
                endIdx = 0
        
        A = A[:, idxKept]
        b = b[idxKept]
        angles = angles[idxKept]
        nrCon = np.sum(idxKept)
        
        # If only one or two halfspaces are left
        if nrCon == 1:
            A = A.T
            return A, b, Ae, be, empty
        elif nrCon == 2:
            A, b, empty = aux_twoHalfspaces(A.T, b, tol)
            return A, b, Ae, be, empty
        
        # 90 degree counter-clockwise shifted halfspaces
        A_ = np.array([A[1, :], -A[0, :]])
        
        # Indices for constraints that are kept
        idxKept = np.ones(nrCon, dtype=bool)
        # Indices for constraints that have been checked
        idxUnchecked = np.ones(nrCon, dtype=bool)
        
        startIdx = 0
        middleIdx = 1
        endIdx = 2
        
        while np.any(idxUnchecked):
            # Halfspace at middleIdx will now be checked
            idxUnchecked[middleIdx] = False
            
            # For potential redundancy, angle between startIdx and endIdx must
            # not be larger than 180 degrees, otherwise intersection at the wrong side
            if aux_anglesTooLarge(angles[startIdx], angles[endIdx]):
                # Increment indices (take care of wrap-around!)
                startIdx, middleIdx, endIdx = aux_incrementIndices(startIdx, middleIdx, endIdx, nrCon)
                continue
            
            # Check if constraint between startIdx and endIdx is necessary by
            # computing the intersection
            p1 = A[:, startIdx] * b[startIdx]
            p2 = A[:, endIdx] * b[endIdx]
            p1vec = A_[:, startIdx]
            p2vec = -A_[:, endIdx]
            
            # If vectors are parallel, then middleIdx non-redundant
            if np.linalg.matrix_rank(np.column_stack([p1vec, p2vec])) < 2:
                startIdx, middleIdx, endIdx = aux_incrementIndices(startIdx, middleIdx, endIdx, nrCon)
                continue
            
            # Solve linear system [p1vec -p2vec] * [a;b] = p2 - p1
            factors = np.linalg.solve(np.column_stack([p1vec, p2vec]), p2 - p1)
            # Plug in
            x = p1 + factors[0] * A_[:, startIdx]
            
            # Evaluate support function of intersecting point and compare to
            # offset of middleIdx'th halfspace
            val = A[:, middleIdx].T @ x
            if b[middleIdx] > val or withinTol(b[middleIdx], val, 1e-9):
                # -> middle halfspace is redundant
                idxKept[middleIdx] = False
                # Increment indices (take care of wrap-around!)
                middleIdx = endIdx
                endIdx = endIdx + 1
                if endIdx >= nrCon:
                    endIdx = 0
            else:
                # -> middle halfspace is non-redundant
                # Increment indices (take care of wrap-around!)
                startIdx, middleIdx, endIdx = aux_incrementIndices(startIdx, middleIdx, endIdx, nrCon)
        
        # Remove halfspaces
        A = A[:, idxKept].T
        b = b[idxKept]
    
    return A, b, Ae, be, empty


def aux_anglesTooLarge(alpha, beta):
    """
    Auxiliary function for 2D case
    Checks if angle between alpha and beta is larger than 180 degrees
    alpha, beta in [-180,180]
    """
    # Shift by 180
    alpha = alpha + 180
    beta = beta + 180
    
    # Shift alpha and beta such that alpha is at 0 (alpha not needed)
    beta = (beta + (360 - alpha)) % 360
    
    # Check if angle is larger than 180
    return beta >= 180


def aux_incrementIndices(startIdx, middleIdx, endIdx, nrCon):
    """
    Auxiliary function for 2D case
    Increment indices by 1; additionally, wrap around at the end of the list
    """
    startIdx = middleIdx
    middleIdx = endIdx
    endIdx = endIdx + 1
    
    # Wrap around if list length is exceeded
    if endIdx >= nrCon:
        endIdx = 0
    
    return startIdx, middleIdx, endIdx


def aux_twoHalfspaces(A, b, tol):
    """
    Polytope with two halfspaces (inequality constraints)
    Note: halfspaces are normalized to unit length
    """
    empty = False
    
    # Compute dot product
    dotprod = A[0, :] @ A[1, :]
    
    # Parallel or anti-parallel?
    if withinTol(dotprod, 1):
        # Remove the one with the shorter offset
        minIdx = np.argmin(b)
        b = np.array([b[minIdx]])
        A = A[minIdx:minIdx+1, :]
    elif withinTol(dotprod, -1) and -b[1] > b[0]:
        empty = True
    
    return A, b, empty 