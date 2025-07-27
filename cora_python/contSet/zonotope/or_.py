"""
or_ - computes an over-approximation for the union of zonotopes

Syntax:
    S_out = Z | S;
    S_out = or_(Z,S)
    S_out = or_(Z,S,alg,order)

Inputs:
    Z - zonotope object
    S - contSet object, numeric, cell-array
    alg - algorithm used to compute the union
               - 'linprog' (default)
               - 'tedrake'
               - 'iterative'
               - 'althoff'
               - 'parallelotope'
    order - zonotope order of the enclosing zonotope

Outputs:
    S_out - zonotope object enclosing the union

Example: 
    Z1 = zonotope([4 2 2;1 2 0])
    Z2 = zonotope([3 1 -1 1;3 1 2 0])
    S_out = Z1 | Z2

    figure; hold on;
    plot(Z1,[1,2],'r')
    plot(Z2,[1,2],'b')
    plot(S_out,[1,2],'g')

References:
    [1] Sadraddini et. al: Linear Encodings for Polytope Containment
        Problems, CDC 2019

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval/or_

Authors:       Matthias Althoff, Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       20-September-2013 (MATLAB)
Last update:   12-November-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, List, Optional, Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from .zonotope import Zonotope
from ..polytope.polytope import Polytope
from ..interval.interval import Interval


def or_(Z: Zonotope, S: Union[Zonotope, np.ndarray, List], 
        alg: str = 'linprog', order: Optional[int] = None) -> Zonotope:
    """
    Computes an over-approximation for the union of zonotopes.
    
    Args:
        Z: Zonotope object
        S: contSet object, numeric, or list of sets
        alg: Algorithm used to compute the union (default: 'linprog')
        order: Zonotope order of the enclosing zonotope (optional)
        
    Returns:
        Zonotope: Object enclosing the union
        
    Raises:
        CORAerror: If inputs are invalid or computation fails
    """
    # Default values
    if alg is None:
        alg = 'linprog'
    
    # Check input arguments
    if not isinstance(Z, (Zonotope, np.ndarray)):
        raise CORAerror('CORA:wrongValue', 'first', 'zonotope or numeric')
    
    if not isinstance(S, (Zonotope, np.ndarray, list)):
        raise CORAerror('CORA:wrongValue', 'second', 'contSet, numeric, or list')
    
    if alg not in ['linprog', 'tedrake', 'iterative', 'althoff', 'parallelotope']:
        raise CORAerror('CORA:wrongValue', 'third', f'Unknown algorithm: {alg}')
    
    if order is not None and not isinstance(order, (int, float)):
        raise CORAerror('CORA:wrongValue', 'fourth', 'numeric')
    
    # Ensure that numeric is second input argument
    Z, S = _reorderNumeric(Z, S)
    
    # Check dimensions
    _equalDimCheck(Z, S)
    
    # Write all into a list
    if isinstance(S, list):
        S_cell = [Z] + S
    else:
        S_cell = [Z, S]
    
    # Only zonotopes, intervals, and numeric allowed
    for s in S_cell:
        if not (isinstance(s, Zonotope) or isinstance(s, Interval) or 
                isinstance(s, (np.ndarray, int, float)) or np.isscalar(s)):
            raise CORAerror('CORA:noops', Z, S)
    
    # Convert all sets to zonotopes, skip empty sets
    S_cell = [Zonotope(s) for s in S_cell]
    S_cell = [s for s in S_cell if not s.representsa_('emptySet', 1e-8)]
    
    # If only one set remains, this is the union
    if len(S_cell) == 0:
        return Zonotope.empty(Z.dim())
    elif len(S_cell) == 1:
        return S_cell[0]
    
    # Compute over-approximation of the union with the selected algorithm
    if alg == 'linprog':
        S_out = _aux_unionLinprog(S_cell, order)
    elif alg == 'althoff':
        S_out = _aux_unionAlthoff(S_cell[0], S_cell[1:], order)
    elif alg == 'tedrake':
        S_out = _aux_unionTedrake(S_cell, order)
    elif alg == 'iterative':
        S_out = _aux_unionIterative(S_cell, order)
    elif alg == 'parallelotope':
        S_out = _aux_unionParallelotope(S_cell)
    else:
        raise CORAerror('CORA:wrongValue', 'third', f'Unknown algorithm: {alg}')
    
    return S_out


def _reorderNumeric(Z, S):
    """Ensure that numeric is second input argument"""
    if isinstance(Z, (np.ndarray, int, float)) or np.isscalar(Z):
        return S, Z
    return Z, S


def _equalDimCheck(Z, S):
    """Check if dimensions are equal"""
    if isinstance(S, list):
        for s in S:
            if hasattr(s, 'dim') and Z.dim() != s.dim():
                raise CORAerror('CORA:dimensionMismatch', Z, s)
    elif hasattr(S, 'dim') and Z.dim() != S.dim():
        raise CORAerror('CORA:dimensionMismatch', Z, S)


def _aux_unionTedrake(Zcell: List[Zonotope], order: Optional[int]) -> Zonotope:
    """Compute the union by solving a linear program with respect to the
    zonotope containment constraints presented in [1]"""
    
    # Construct generator matrix of the final zonotope
    Z_ = _aux_unionIterative(Zcell, order)
    G = Z_.generators
    
    Y = G @ np.diag(1/np.sqrt(np.sum(G**2, axis=0)))
    n, ny = Y.shape
    Hy = np.vstack([np.eye(ny), -np.eye(ny)])
    
    # Construct linear constraints for each zonotope
    Aeq = None
    beq = None
    A = None
    A_ = None
    
    for i in range(len(Zcell)):
        # Obtain generator matrix and center from the current zonotope
        X = Zcell[i].generators
        x = Zcell[i].center
        nx = X.shape[1]
        Hx = np.vstack([np.eye(nx), -np.eye(nx)])
        hx = np.ones((2*nx, 1))
        
        # Construct constraint X = Y * T
        temp = [Y] * nx
        A1 = np.hstack([np.block([[temp[j] if j == k else np.zeros_like(Y) for k in range(nx)] for j in range(nx)]),
                        np.zeros((n*nx, 4*nx*ny)),
                        np.zeros((n*nx, ny))])
        b1 = X.flatten().reshape(-1, 1)
        
        # Construct constraint x = Y * beta
        A2 = np.hstack([np.zeros((n, nx*ny)), np.zeros((n, 4*nx*ny)), Y])
        b2 = -x
        
        # Construct constraint lambda * Hx = Hy * T
        Atemp = []
        for j in range(Hx.shape[1]):
            h = Hx[:, j:j+1]
            temp = [h.T] * Hy.shape[0]
            Atemp.append(np.block([[temp[k] if k == l else np.zeros_like(h.T) for l in range(Hy.shape[0])] for k in range(len(temp))]))
        
        temp = [Hy] * Hx.shape[1]
        A3 = np.hstack([np.block([[temp[j] if j == k else np.zeros_like(Hy) for k in range(Hx.shape[1])] for j in range(len(temp))]),
                        -np.vstack(Atemp),
                        np.zeros((len(Atemp), ny))])
        b3 = np.zeros((A3.shape[0], 1))
        
        # Add current equality constraint to overall equality constraints
        Atemp = np.vstack([A1, A2, A3])
        btemp = np.vstack([b1, b2, b3])
        
        if Aeq is None:
            Aeq = Atemp
            beq = btemp
        else:
            Aeq = np.block([[Aeq, np.zeros((Aeq.shape[0], Atemp.shape[1]))],
                           [np.zeros((Atemp.shape[0], Aeq.shape[1])), Atemp]])
            beq = np.vstack([beq, btemp])
        
        # Construct constraint lambda * hx <= hy + Hy beta
        temp = [hx.T] * Hy.shape[0]
        if A_ is None:
            A_ = -np.eye(Hy.shape[0])
        else:
            A_ = np.block([[A_, np.zeros((A_.shape[0], Hy.shape[0]))],
                          [np.zeros((Hy.shape[0], A_.shape[1])), -np.eye(Hy.shape[0])]])
        
        A1 = np.hstack([np.zeros((Hy.shape[0], Y.shape[1]*X.shape[1])),
                        np.block([[temp[k] if k == l else np.zeros_like(hx.T) for l in range(Hy.shape[0])] for k in range(len(temp))]),
                        -Hy])
        
        # Construct constraint lambda >= 0
        A2 = np.hstack([np.zeros((4*nx*ny, nx*ny)), -np.eye(4*nx*ny), np.zeros((4*nx*ny, ny))])
        if A_ is None:
            A_ = np.zeros((4*nx*ny, 2*ny))
        else:
            A_ = np.block([[A_, np.zeros((A_.shape[0], 2*ny))],
                          [np.zeros((4*nx*ny, A_.shape[1])), np.zeros((4*nx*ny, 2*ny))]])
        
        if A is None:
            A = np.vstack([A1, A2])
        else:
            A = np.block([[A, np.zeros((A.shape[0], A1.shape[1] + A2.shape[1]))],
                         [np.zeros((A1.shape[0] + A2.shape[0], A.shape[1])), np.vstack([A1, A2])]])
    
    # Solve linear program
    f = np.concatenate([np.ones(2*ny), np.zeros(Aeq.shape[1] if Aeq is not None else 0)])
    
    Atemp = np.hstack([-np.eye(ny), -np.eye(ny)])
    if A is not None:
        A = np.vstack([np.hstack([A_, A]), np.hstack([Atemp, np.zeros((ny, A.shape[1]))])])
    else:
        A = np.vstack([A_, np.hstack([Atemp, np.zeros((ny, 0))])])
    
    b = np.zeros((A.shape[0], 1))
    
    if Aeq is not None:
        Aeq = np.hstack([np.zeros((Aeq.shape[0], 2*ny)), Aeq])
    
    problem = {
        'f': f,
        'Aineq': A,
        'bineq': b,
        'Aeq': Aeq,
        'beq': beq,
        'lb': None,
        'ub': None
    }
    
    val = CORAlinprog(problem)[0]
    
    # Construct the resulting zonotope
    ub = val[:ny]
    lb = -val[ny:2*ny]
    int_val = Interval(lb, ub)
    
    c = Y @ int_val.center
    G = Y @ np.diag(int_val.rad().flatten())
    
    return Zonotope(c, G)


def _aux_unionLinprog(Zcell: List[Zonotope], order: Optional[int]) -> Zonotope:
    """Compute an enclosing zonotope using linear programming. As the
    constraints for the linear program we compute the upper and lower bound
    for all zonotopes that are enclosed in the normal directions of the
    halfspace representation of the enclosing zonotope"""
    
    # Construct generator matrix of the final zonotope
    nrZ = len(Zcell)
    Z_ = _aux_unionIterative(Zcell, order)
    Z_ = Z_.compact_('zeros', 1e-8)
    G = Z_.generators()
    
    G = G @ np.diag(1/np.sqrt(np.sum(G**2, axis=0)))
    
    # Compute the directions of the boundary halfspaces
    n, nrGen = G.shape
    P = Polytope(Z_ - Z_.center)
    nrIneq = len(P.b)
    
    val = np.zeros((nrIneq, nrZ))
    for i in range(nrIneq):
        # Loop over all zonotopes
        for j in range(nrZ):
            # Compute bound for the current zonotope (note: this is a
            # direct implementation of zonotope/supportFunc_ ...)
            Z_proj = P.A[i:i+1, :] @ Zcell[j]
            val[i, j] = Z_proj.center[0, 0] + np.sum(np.abs(Z_proj.generators()))
    
    d = np.max(val, axis=1)
    
    # Solve linear program
    f = np.concatenate([np.zeros(n), np.ones(nrGen)])
    
    problem = {
        'f': f,
        'Aineq': np.vstack([np.hstack([-P.A, -np.abs(P.A @ G)]),
                            np.hstack([np.zeros((nrGen, n)), -np.eye(nrGen)])]),
        'bineq': np.vstack([-d.reshape(-1, 1), np.zeros((nrGen, 1))]),
        'Aeq': None,
        'beq': None,
        'lb': None,
        'ub': None
    }
    
    result = CORAlinprog(problem)
    x = result[0]  # Get first return value like MATLAB
    
    if x is None:
        raise CORAerror('CORA:solverIssue', 'Linear programming solver failed')
    
    # Ensure x is a 1D array
    x = x.flatten()
    
    # Construct final zonotope
    c = x[:n].reshape(-1, 1)
    scal = x[n:]
    
    return Zonotope(c, G @ np.diag(scal))


def _aux_unionIterative(Zcell: List[Zonotope], order: Optional[int]) -> Zonotope:
    """Iteratively unite all zonotopes with the enclose function"""
    
    Zcell_ = [None] * len(Zcell)
    
    # Loop until all sets have been united
    while len(Zcell) > 1:
        counter = 0
        counter_ = 0
        
        # Loop over all sets in the current queue
        while counter < len(Zcell) - 1:
            # Unite sets using function enclose
            from .enclose import enclose
            Zcell_[counter_] = enclose(Zcell[counter], Zcell[counter + 1])
            
            counter_ += 1
            counter += 2
        
        # Consider remaining set
        if counter == len(Zcell) - 1:
            Zcell_[counter_] = Zcell[counter]
            counter_ += 1
        
        Zcell = Zcell_[:counter_]
    
    # Reduce the resulting zonotope to the desired order
    Z = Zcell[0]
    
    if order is not None:
        if order == 1:
            Z = Z.reduce_('methC', order)
        else:
            Z = Z.reduce_('pca', order)
    
    return Z


def _aux_unionAlthoff(Z1: Zonotope, Zcell: List[Zonotope], order: Optional[int]) -> Zonotope:
    """Althoff's union algorithm"""
    
    # Init
    Zmat = []
    
    # Dimension
    n = Z1.dim()
    
    # Obtain minimum number of generators every zonotope has
    minNrOfGens = Z1.generators().shape[1]
    for iSet in range(len(Zcell)):
        minNrOfGens = min(minNrOfGens, Zcell[iSet].generators().shape[1])
    
    # Obtain Zcut
    Zcut = [Z1.generators()[:, :minNrOfGens]]
    for iSet in range(len(Zcell)):
        Zcut.append(np.hstack([Zcell[iSet].center, Zcell[iSet].generators()[:, :minNrOfGens]]))
    
    # Obtain Zadd
    Zadd = [Z1.generators()[:, minNrOfGens:]]
    for iSet in range(len(Zcell)):
        Zadd.append(Zcell[iSet].generators()[:, minNrOfGens:])
    
    # As center is prepended
    minNrOfGens = minNrOfGens + 1
    
    # Compute vertex sets for each set of generators
    for iGen in range(minNrOfGens):
        v = Zcut[0][:, iGen:iGen+1]
        for iSet in range(1, len(Zcut)):
            v_new = Zcut[iSet][:, iGen:iGen+1]
            # If direction is correct
            if v.T @ v_new > 0:
                v = np.hstack([v, v_new])
            # Flip direction
            else:
                v = np.hstack([v, -v_new])
        
        # Compute vertices - v is already a matrix of points
        V = v
        
        # Compute enclosing zonotope
        Z_encl = Zonotope.enclosePoints(V)
        
        # Concatenate enclosing zonotopes
        Zmat = np.hstack([Zmat, np.hstack([Z_encl.center, Z_encl.generators()])])
    
    # Add Zadd to the resulting generator matrix
    for i in range(len(Zadd)):
        Zmat = np.hstack([Zmat, Zadd[i]])
    
    # Create enclosing zonotope
    Z = Zonotope(Zmat)
    
        # Reduce zonotope to the desired zonotope order
    if order is not None:
       Z = Z.reduce_('pca', order)
    
    return Z


def _aux_unionParallelotope(Zcell: List[Zonotope]) -> Zonotope:
    """Enclose the union of the zonotopes with a parallelotope"""
    
    # Obtain matrix of points from generator matrices
    n = Zcell[0].dim()
    V = np.zeros((n, 0))
    for i in range(len(Zcell)):
        G = Zcell[i].generators
        V = np.hstack([V, G, -G])
    
    # Compute the arithmetic mean of the vertices
    meanV = np.sum(V, axis=1) / n
    
    # Obtain sampling matrix
    sampleMatrix = V - meanV.reshape(-1, 1)
    
    # Compute the covariance matrix
    C = np.cov(sampleMatrix.T)
    
    # Singular value decomposition
    U, _, _ = np.linalg.svd(C)
    
    # Enclose zonotopes with intervals in the transformed space
    from .interval import interval
    I = Interval.empty(n)
    for i in range(len(Zcell)):
        transformed_zono = U.T @ Zcell[i]
        I = interval(transformed_zono) | I
    
    # Transform back to original space
    return U @ Zonotope(I) 