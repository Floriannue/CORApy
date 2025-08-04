"""
simplex - enclose a zonotope by a simplex

Syntax:
    P = simplex(Z)

Inputs:
    Z - zonotope object

Outputs:
    P - polytope object representing the simplex

Example: 
    Z = Zonotope(np.array([[1], [0]]), np.array([[1, -1, 0.5], [0, 1, 1]]))
    P = simplex(Z)
    
    # Visualization (equivalent to MATLAB plotting)
    # figure; hold on; box on;
    # plot(Z);
    # plot(P, [1,2], 'r');

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/interval, zonotope/polytope

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       31-May-2022 (MATLAB)
Last update:   --- (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def simplex(Z: Zonotope):
    """
    Enclose a zonotope by a simplex.
    
    Args:
        Z: zonotope object
        
    Returns:
        P: polytope object representing the simplex
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Get dimension
    n = Z.c.shape[0]
    
    # Construct an n-dimensional standard simplex with origin 0
    V = np.eye(n + 1)
    
    # Create basis matrix for Gram-Schmidt
    # In MATLAB: if size(B,2) == 1, B_ = eye(length(B)); [~,ind] = max(abs(B'*B_)); B_(:,ind) = []; B = [B,B_];
    v = np.ones((n + 1, 1))
    B_ = np.eye(n + 1)
    # Find the index of maximum absolute value in v'*B_
    dot_products = np.abs(v.T @ B_)
    ind = np.argmax(dot_products)
    # Remove the column at index ind
    B_ = np.delete(B_, ind, axis=1)
    # Concatenate v and B_
    B_input = np.hstack([v, B_])
    
    from cora_python.g.functions.matlab.init.gramSchmidt import gramSchmidt
    B = gramSchmidt(B_input)
    
    # Create polytope from the simplex vertices
    # (B[:,2:end]'*V) creates vertices for the simplex
    vertices = B[:, 1:].T @ V
    from cora_python.contSet.polytope import Polytope
    P = Polytope(vertices)
    
    # Compute the halfspace representation
    from cora_python.contSet.polytope.constraints import constraints
    P = constraints(P)
    
    # Scale the simplex so that it tightly encloses the zonotope
    A = P.A
    if A is None or A.size == 0:
        raise CORAerror('CORA:wrongInputInConstructor', 'Polytope constraints are None or empty')
    
    # Compute supremum of interval(A*Z)
    # First compute A*Z (matrix multiplication of A with zonotope Z)
    from cora_python.contSet.zonotope.mtimes import mtimes
    AZ = mtimes(A, Z)
    
    # Convert to interval and get supremum
    from cora_python.contSet.zonotope.interval import interval
    AZ_interval = interval(AZ)
    if AZ_interval is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Could not create interval from A*Z')
    
    b = AZ_interval.supremum()
    if b is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Interval supremum is None')
    
    # Create final polytope
    P = Polytope(A, b)
    
    return P


 