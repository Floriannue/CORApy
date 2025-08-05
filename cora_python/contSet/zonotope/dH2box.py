"""
dH2box - computes an over-approximation of the Hausdorff distance
    to the interval over-approximation of the provided zonotope Z
    note: this function is not optimized w.r.t computational efficiency

Syntax:
    dH = dH2box(Z, method)

Inputs:
    Z - zonotope object
    method - char array of over-approximation method:
       - 'exact' [eq.(5), 1]
       - 'naive': radius of box over-approximation
       - 'ell': sequence of parallelotopes P_i, each with inscribed
                   ball containing box (= under-approximation of P_i)
       - 'wgreedy' [Thm.3.2, 2]
       - 'wopt': modification of [Thm.3.2, 2]

Outputs:
    dH - over-approximated Hausdorff distance

References:
    [1] X. Yang, J.K. Scott. A comparison of zonotope order reduction
        techniques. Automatica 95 (2018), pp. 378-384.
    [2] M. Wetzlinger, A. Kulmburg, M. Althoff. Adaptive parameter tuning
        for reachability analysis of nonlinear systems. HSCC 2021.
    [3] V. Gassmann, M. Althoff. Scalable zonotope-ellipsoid conversions
        using the euclidean zonotope norm. ACC 2020.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also:

Example:
    Z = Zonotope(np.array([[0], [0]]), -1 + 2 * np.random.rand(2, 20))
    dH = dH2box(Z, 'naive')

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 08-March-2021 (MATLAB)
Last update: --- (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from .interval import interval


def dH2box(Z: Zonotope, method: Optional[str] = None) -> float:
    """
    Computes an over-approximation of the Hausdorff distance to the interval 
    over-approximation of the provided zonotope Z
    
    Args:
        Z: zonotope object
        method: over-approximation method ('exact', 'naive', 'ell', 'wgreedy', 'wopt')
        
    Returns:
        Over-approximated Hausdorff distance
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.random.rand(2, 20) * 2 - 1)
        dH = dH2box(Z, 'naive')
    """
    # Input arguments
    if Z.dim() <= 3:
        dV = 'exact'
    else:
        dV = 'naive'
    
    # Parse input arguments
    if method is None:
        method = dV
    
    # Check input arguments
    inputArgsCheck([
        [Z, 'att', 'zonotope'],
        [method, 'str', ['exact', 'naive', 'ell', 'wgreedy', 'wopt']]
    ])
    
    # Methods
    if method == 'exact':
        dH = _aux_dH2box_exact(Z)
    elif method == 'naive':
        dH = _aux_dH2box_naive(Z)
    elif method == 'ell':
        dH = _aux_dH2box_ell(Z)
    elif method == 'wgreedy':
        dH = _aux_dH2box_wgreedy(Z)
    elif method == 'wopt':
        dH = _aux_dH2box_wopt(Z)
    
    return dH


def _aux_dH2box_exact(Z: Zonotope) -> float:
    """
    Computes near-exact dH according to [1]
    """
    # Generator matrices
    G = Z.generators()
    Gbox = np.diag(np.sum(np.abs(G), axis=1))
    
    # Generate lambda vectors
    n = Z.dim()
    numDirs = 10000
    sections = int(np.power(numDirs, 1/(n-1))) if n > 1 else numDirs
    lambda_vectors = _randEqdistDirections(n, sections)
    
    # Loop over each lambda to find maximum
    dH = 0.0
    for i in range(lambda_vectors.shape[1]):
        lambda_vec = lambda_vectors[:, i:i+1]
        dHlambda = abs(float(np.linalg.norm(lambda_vec.T @ Gbox, 1)) - 
                       float(np.linalg.norm(lambda_vec.T @ G, 1)))
        if dHlambda > dH:
            dH = dHlambda
    
    return float(dH)


def _aux_dH2box_naive(Z: Zonotope) -> float:
    """
    Computes the radius of the box over-approximation of Z
    """
    # Use vecnorm(rad(interval(Z))) as in MATLAB
    I = interval(Z)
    radius_vec = I.rad()  # Get radius vector
    return float(np.linalg.norm(radius_vec, 2))  # vecnorm equivalent


def _aux_dH2box_ell(Z: Zonotope) -> float:
    """
    Computes distance from box under-approximations to box over-approximations
    """
    # Generator matrix
    G = Z.generators()
    n, gamma = G.shape
    
    # Number of parallelotopes
    nrP = int(np.floor(gamma / n))
    
    # Take all generators for now (simplified)
    GforP = G.copy()
    
    # Container for Hausdorff distances
    dH = np.zeros(nrP)
    
    for i in range(nrP):
        # Extract invertible parallelotope P (simplified)
        if GforP.shape[1] < n:
            break
        
        # Take first n generators as parallelotope
        P = GforP[:, :n]
        GforP = GforP[:, n:]
        
        # Vertex of interval over-approximation of P
        vPover = np.sum(np.abs(P), axis=1)
        
        # Invert generator matrix: A*x <= 1 is h-rep of P
        try:
            A = np.linalg.inv(P)
        except np.linalg.LinAlgError:
            # If P is not invertible, skip this parallelotope
            continue
        
        # Normalize rows of A and b
        c = 1.0 / np.sqrt(np.sum(A**2, axis=1))
        bnorm = c
        
        # Radius of inscribed ball
        rball = np.min(np.abs(bnorm))
        
        # Radius of box in inscribed ball
        rBox = np.sqrt(rball**2 / n)
        
        # Vertex of interval under-approximation of P
        vPunder = rBox * np.ones(n)
        
        # Measure Hausdorff distance
        dH[i] = np.linalg.norm(vPover - vPunder, 2)
    
    # Accumulated Hausdorff distance
    return float(np.sum(dH))


def _aux_dH2box_wgreedy(Z: Zonotope) -> float:
    """
    Computes length of sum of generators with largest absolute element set to 0
    """
    # Generator matrices
    G = Z.generators()
    Gabs = np.abs(G)
    _, gamma = G.shape
    
    # Truncated generators (largest absolute element per column = 0)
    Gtruncated = Gabs.copy()
    for k in range(gamma):
        idxmax = np.argmax(Gabs[:, k])
        Gtruncated[idxmax, k] = 0
    
    # Measure
    dH = 2 * np.linalg.norm(np.sum(Gtruncated, axis=1), 2)
    
    return float(dH)


def _aux_dH2box_wopt(Z: Zonotope) -> float:
    """
    Computes optimal scaling of generator lengths
    """
    # Generator matrices
    G = Z.generators()
    Gabs = np.abs(G)
    
    # Dimensions
    n, gamma = G.shape
    
    # Lengths of generators
    lsquared = np.linalg.norm(G, 2, axis=0) ** 2
    
    dH = 0.0
    for i in range(n):
        summand = 0.0
        for k in range(gamma):
            if lsquared[k] != 0:
                summand += Gabs[i, k] * (1 - G[i, k]**2 / lsquared[k])
        dH += summand**2
    
    dH = 2 * np.sqrt(dH)
    
    return float(dH)


def _randEqdistDirections(n: int, sections: int) -> np.ndarray:
    """
    Generate equally distributed directions (simplified version)
    """
    # Simplified implementation - generate random directions
    numDirs = min(10000, sections**(n-1))
    lambda_vectors = np.random.randn(n, numDirs)
    lambda_vectors = lambda_vectors / np.linalg.norm(lambda_vectors, axis=0)
    return lambda_vectors 