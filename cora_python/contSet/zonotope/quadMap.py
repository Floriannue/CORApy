"""
quadMap - computes the quadratic map of a zonotope

Description:
    This function computes the quadratic map of a zonotope according to the 
    methods described in [1].

Syntax:
    Zquad = quadMap(Z1, Q)
    Zquad = quadMap(Z1, Z2, Q)

Inputs:
    Z1 - zonotope object
    Z2 - zonotope object (optional)
    Q - quadratic coefficients as a list of matrices

Outputs:
    Zquad - zonotope object

Example:
    Z = zonotope([0, 1, 1; 0, 1, 0])
    Q = [[[0.5, 0.5], [0, -0.5]], [[-1, 0], [1, 1]]]
    res = quadMap(Z, Q)

References:
    [1] M. Althoff et al. "Avoiding Geometric Intersection Operations in 
        Reachability Analysis of Hybrid Systems", HSCC 2011.

Authors: Matthias Althoff, Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 07-December-2011 (MATLAB)
Last update: 22-November-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import List
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.helper.sets.contSet.zonotope import nonzeroFilter

from .zonotope import Zonotope

def quadMap(Z1: Zonotope, *args) -> Zonotope:
    """
    Compute the quadratic map of a zonotope
    
    Args:
        Z1: First zonotope object
        *args: Either (Q,) for single zonotope case or (Z2, Q) for mixed case
        
    Returns:
        Zonotope object representing the quadratic map
    """
    
    if len(args) == 1:
        Q = args[0]
        # Check if Q contains matZonotope objects (not implemented yet)
        # For now, assume regular matrices
        return _aux_quadMapSingle(Z1, Q)
    elif len(args) == 2:
        Z2, Q = args
        return _aux_quadMapMixed(Z1, Z2, Q)
    else:
        raise ValueError("Invalid number of input arguments")


def _aux_quadMapSingle(Z: Zonotope, Q: List[np.ndarray]) -> Zonotope:
    """
    Compute an over-approximation of the quadratic map 
    {x_i = x^T Q{i} x | x ∈ Z} 
    of a zonotope according to Lemma 1 in [1]
    """
    
    # Get matrix of zonotope [c, G]
    if Z.G.size == 0:
        Zmat = Z.c
    else:
        Zmat = np.hstack([Z.c, Z.G])
    
    dimQ = len(Q)
    gens = Z.G.shape[1] if Z.G.size > 0 else 0

    # Initialize solution
    c = np.zeros((dimQ, 1))
    G = np.zeros((dimQ, int(0.5 * (gens**2 + gens) + gens)))
    
    # Count non-empty matrices
    Qnonempty = np.zeros(dimQ, dtype=bool)
    
    # For each dimension, compute generator elements
    for i in range(dimQ):
        Q_i = np.asarray(Q[i])
        Qnonempty[i] = np.any(Q_i)
        
        if Qnonempty[i]:
            # Pure quadratic evaluation
            quadMat = Zmat.T @ Q_i @ Zmat
            
            if gens > 0:
                # Diagonal elements
                G[i, :gens] = 0.5 * np.diag(quadMat[1:gens+1, 1:gens+1])
                
                # Center
                c[i, 0] = quadMat[0, 0] + np.sum(G[i, :gens])
                
                # Off-diagonal elements
                quadMatoffdiag = quadMat + quadMat.T
                quadMatoffdiag_flat = quadMatoffdiag.flatten()
                
                # Create lower triangular mask (excluding diagonal)
                kInd = np.tril(np.ones((gens+1, gens+1), dtype=bool), -1)
                G[i, gens:] = quadMatoffdiag_flat[kInd.flatten()]
            else:
                # No generators case
                c[i, 0] = quadMat[0, 0] if quadMat.size > 0 else 0
    
    # Generate new zonotope
    tmp_sum = np.sum(Qnonempty)
    if tmp_sum < 1 or withinTol(tmp_sum, 1):
        # Single or no non-empty Q matrices
        G_sum = np.sum(np.abs(G), axis=1, keepdims=True)
        return Zonotope(c, G_sum)
    else:
        # Multiple non-empty Q matrices
        G_filtered = nonzeroFilter(G)
        return Zonotope(c, G_filtered)


def _aux_quadMapMixed(Z1: Zonotope, Z2: Zonotope, Q: List[np.ndarray]) -> Zonotope:
    """
    Compute an over-approximation of the quadratic map 
    {x_i = x1^T Q{i} x2 | x1 ∈ Z1, x2 ∈ Z2} 
    of two zonotope objects.
    """
    
    # Get matrices of zonotopes [c, G]
    if Z1.G.size == 0:
        Zmat1 = Z1.c
    else:
        Zmat1 = np.hstack([Z1.c, Z1.G])
        
    if Z2.G.size == 0:
        Zmat2 = Z2.c
    else:
        Zmat2 = np.hstack([Z2.c, Z2.G])
    
    dimQ = len(Q)
    
    # Initialize solution (center + generator matrix)
    Z = np.zeros((dimQ, Zmat1.shape[1] * Zmat2.shape[1]))
    
    # Count non-empty matrices
    Qnonempty = np.zeros(dimQ, dtype=bool)

    # For each dimension, compute center + generator elements
    for i in range(dimQ):
        Q_i = np.asarray(Q[i])
        Qnonempty[i] = np.any(Q_i)
        
        if Qnonempty[i]:
            # Pure quadratic evaluation
            quadMat = Zmat1.T @ Q_i @ Zmat2
            Z[i, :] = quadMat.flatten()

    # Generate new zonotope
    tmp_sum = np.sum(Qnonempty)
    if tmp_sum < 1 or withinTol(tmp_sum, 1):
        # Single or no non-empty Q matrices
        c = Z[:, 0:1]  # First column as center
        G_sum = np.sum(np.abs(Z[:, 1:]), axis=1, keepdims=True)
        return Zonotope(c, G_sum)
    else:
        # Multiple non-empty Q matrices
        c = Z[:, 0:1]  # First column as center
        G_filtered = nonzeroFilter(Z[:, 1:])
        return Zonotope(c, G_filtered) 