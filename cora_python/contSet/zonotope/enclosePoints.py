"""
enclosePoints - enclose a point cloud with a zonotope

This method implements two algorithms to enclose a point cloud with a zonotope:
- 'stursberg': Method from Stursberg et al. using SVD and oriented bounding box
- 'maiga': Method from Maiga et al. using iterative optimization

Syntax:
    Z = Zonotope.enclosePoints(points)
    Z = Zonotope.enclosePoints(points, method)

Inputs:
    points - matrix storing point cloud (dimension: [n,p] for p points)
    method - method used for calculation
               - 'maiga' (default)
               - 'stursberg'

Outputs:
    Z - Zonotope object

Example: 
    points = -1 + 2*np.random.rand(2,10)

    Z1 = Zonotope.enclosePoints(points)
    Z2 = Zonotope.enclosePoints(points,'maiga')
    
    # figure; hold on
    # plt.plot(points[0,:],points[1,:],'k.')
    # Z1.plot([1,2],'r')
    # Z2.plot([1,2],'b')

References:
    [1] O. Stursberg et al. "Efficient representation and computation of 
        reachable sets for hybrid systems", HSCC 2003
    [2] M. Maiga et al. "A Comprehensive Method for Reachability Analysis
        of Uncertain Nonlinear Hybrid Systems", TAC 2017

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np
from typing import TYPE_CHECKING, Union, Optional

from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval

def enclosePoints(points: np.ndarray, method: str = 'maiga') -> Zonotope:
    """
    enclosePoints - enclose a point cloud with a zonotope
    
    Args:
        points: matrix storing point cloud (dimension: [n,p] for p points)
        method: method used for calculation ('maiga' or 'stursberg')
        
    Returns:
        Zonotope object that encloses all the points
        
    Raises:
        ValueError: If invalid method or empty points
    """
    
    # Convert to numpy array
    points = np.array(points, dtype=float)
    
    # Check input arguments - simplified validation since inputArgsCheck seems to have issues
    if not isinstance(points, np.ndarray):
        points = np.array(points, dtype=float)
    
    if method not in ['maiga', 'stursberg']:
        raise ValueError(f"Method must be 'maiga' or 'stursberg', got: {method}")
    
    # Handle edge cases
    if points.size == 0:
        raise ValueError("Empty point cloud cannot be enclosed")
    
    # Ensure 2D array
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    
    # Compute enclosing zonotope with the selected method
    if method == 'stursberg':
        return _enclose_points_stursberg(points)
    elif method == 'maiga':
        return _enclose_points_maiga(points)
    else:
        raise ValueError(f"Unknown method: {method}")


def _enclose_points_stursberg(points: np.ndarray) -> Zonotope:
    """
    Computes an enclosing zonotope using the method from [1]
    
    Args:
        points: Point cloud matrix [n,p]
        
    Returns:
        Zonotope object
    """
    
    # Compute the arithmetic mean of the points
    mean = np.mean(points, axis=1, keepdims=True)
    
    # Obtain sampling matrix
    sample_matrix = points - mean
    
    # Compute the covariance matrix
    C = np.cov(sample_matrix.T, rowvar=False)
    
    # Handle 1D case
    if C.ndim == 0:
        C = np.array([[C]])
    elif C.ndim == 1:
        C = np.diag(C)
    
    # Singular value decomposition
    U, _, _ = np.linalg.svd(C)
    
    # Auxiliary computations
    oriented_matrix = U.T @ sample_matrix
    m1 = np.max(oriented_matrix, axis=1)
    m2 = np.min(oriented_matrix, axis=1)
    
    # Determine the center
    c = mean.flatten() + U @ ((m1 + m2) / 2)
    
    # Determine the generators
    G = np.zeros((len(c), len(m1)))
    for i in range(len(m1)):
        G[:, i] = (m1[i] - m2[i]) * 0.5 * U[:, i]
    
    # Remove zero generators
    nonzero_cols = np.any(G != 0, axis=0)
    if np.any(nonzero_cols):
        G = G[:, nonzero_cols]
    else:
        G = np.zeros((len(c), 0))
    
    return Zonotope(np.column_stack([c, G]))


def _enclose_points_maiga(points: np.ndarray) -> Zonotope:
    """
    Computes an enclosing zonotope using the method from [2]
    
    Args:
        points: Point cloud matrix [n,p]
        
    Returns:
        Zonotope object
    """
    
    # Initialization
    min_vol = np.inf
    Z_opt = None
    N = 100
    
    # Loop over all sampling points
    for i in range(1, N + 1):
        # Compute current ratio
        r = i / N
        
        # Compute enclosing zonotope
        Z = _cloud2zonotope(points, r, points.shape[0])
        
        # Estimate the volume of the zonotope using the trace
        G = Z.G
        tr = np.trace(G.T @ G)
        
        # Update the minimum value
        if tr < min_vol:
            Z_opt = Z
            min_vol = tr
    
    return Z_opt


def _cloud2zonotope(X: np.ndarray, ratio: float, s1: int) -> Zonotope:
    """
    Implementation of the cloud2zonotope function in [2]
    
    Args:
        X: Point cloud matrix
        ratio: Compression ratio
        s1: Number of iterations
        
    Returns:
        Zonotope object
    """
    
    n = X.shape[0]
    c = np.zeros(n)
    iter_count = 0
    R = np.zeros((n, 0))
    
    while iter_count < s1:
        iter_count += 1
        
        # Enclose points with interval
        I = Interval.enclosePoints(X)
        mid = I.center()
        r = I.rad()
        
        X = X - mid.reshape(-1, 1)
        
        # SVD decomposition
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        if U.size > 0:
            u = U[:, 0]
        else:
            u = np.zeros(n)
        
        # Compute generator
        g = ratio * abs(np.dot(u, r)) * u
        
        # Compress point cloud
        X = _compress(X, g)
        
        # Update center and generators
        c = c + mid
        if g.size > 0:
            R = np.column_stack([R, g])
    
    # Final interval enclosure
    I = Interval.enclosePoints(X)
    mid = I.center()
    r = I.rad()
    c = c + mid
    
    # Add final generators as diagonal matrix
    R = np.column_stack([R, np.diag(r)])
    
    return Zonotope(np.column_stack([c, R]))


def _compress(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Implementation of the compress function in [2]
    
    Args:
        X: Point cloud matrix
        g: Generator vector
        
    Returns:
        Compressed point cloud
    """
    if np.linalg.norm(g) == 0:
        return X
    
    u = g / np.linalg.norm(g)
    
    for i in range(X.shape[1]):
        d = np.dot(X[:, i], u)
        d = min(np.linalg.norm(g), max(-np.linalg.norm(g), d))
        X[:, i] = X[:, i] - (d * u)
    
    return X 