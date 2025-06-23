"""
plus - overloaded '+' operator for the Minkowski addition of a zonotope
    with another set or vector

Syntax:
    S_out = Z + S
    S_out = plus(Z, S)

Inputs:
    Z - zonotope object, numeric
    S - contSet object, numeric

Outputs:
    S_out - zonotope after Minkowski addition

Example:
    Z = zonotope([1, 0], [[1, 0], [0, 1]])
    Z1 = Z + Z
    Z2 = Z + [2, -1]

References:
    [1] M. Althoff. "Reachability analysis and its application to the 
        safety assessment of autonomous cars", 2010

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2006 (MATLAB)
Last update:   25-February-2025 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .zonotope import Zonotope

def plus(Z, S):
    """
    Overloaded '+' operator for Minkowski addition of zonotope with another set or vector
    
    Args:
        Z: zonotope object or numeric
        S: contSet object or numeric
        
    Returns:
        zonotope: Result of Minkowski addition
        
    Raises:
        CORAerror: If operation is not supported or dimensions don't match
    """
    
    # Ensure that numeric is second input argument (reorder if necessary)
    S_out, S = _reorder_numeric(Z, S)
    
    # Call function with higher precedence if applicable
    if hasattr(S, 'precedence') and hasattr(S_out, 'precedence') and S.precedence > S_out.precedence:
        return S + S_out
    
    try:
        # Different cases depending on the class of the second summand
        # Check for zonotope using both isinstance and class name for robustness
        # (handles different import paths that might create different class instances)
        if isinstance(S, Zonotope) or (hasattr(S, '__class__') and S.__class__.__name__ == 'Zonotope'):
            # Zonotope + Zonotope: see Equation 2.1 in [1]
            new_c = S_out.c + S.c
            if new_c.size == 0:
                return Zonotope.empty(S_out.dim())
            
            # Concatenate generator matrices
            if S_out.G.size > 0 and S.G.size > 0:
                new_G = np.hstack([S_out.G, S.G])
            elif S_out.G.size > 0:
                new_G = S_out.G
            elif S.G.size > 0:
                new_G = S.G
            else:
                # Both S_out.G and S.G are empty
                new_G = np.array([]).reshape(new_c.shape[0], 0)
            
            return Zonotope(new_c, new_G)
        
        # Numeric has to be a scalar or a column vector of correct size
        if isinstance(S, (int, float, np.number)):
            # Scalar addition
            new_c = S_out.c + S
            return Zonotope(new_c, S_out.G)
        elif isinstance(S, (list, tuple, np.ndarray)):
            S_vec = np.asarray(S)
            # Ensure S_vec is a column vector if it's meant to be added to the center
            if S_vec.ndim == 1:
                S_vec = S_vec.reshape(-1, 1)
            elif S_vec.ndim == 2 and S_vec.shape[1] != 1:
                S_vec = S_vec.flatten().reshape(-1, 1) # Flatten and convert to column vector
            
            if S_vec.shape[0] == S_out.c.shape[0]:
                # Vector addition
                new_c = S_out.c + S_vec
                return Zonotope(new_c, S_out.G)
            else:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Dimension mismatch in addition')
        
        # Handle interval case - convert interval to zonotope first
        if hasattr(S, '__class__') and S.__class__.__name__ == 'Interval':
            # Convert interval to zonotope: center = interval center, generators = diag(radius)
            
            S_center = S.center()
            S_radius = S.rad()
            
            # Create generator matrix as diagonal matrix of radii
            # Remove zero generators (where radius is 0)
            nonzero_indices = S_radius.flatten() != 0
            if np.any(nonzero_indices):
                S_G = np.diag(S_radius.flatten())[:, nonzero_indices]
            else:
                S_G = np.zeros((len(S_center), 0))
            
            # Create zonotope from interval
            S_zono = Zonotope(S_center, S_G)
            
            # Add zonotopes
            new_c = S_out.c + S_zono.c
            if new_c.size == 0:
                return Zonotope.empty(S_out.dim())
            
            # Concatenate generator matrices
            if S_out.G.size > 0 and S_zono.G.size > 0:
                new_G = np.hstack([S_out.G, S_zono.G])
            elif S_out.G.size > 0:
                new_G = S_out.G
            elif S_zono.G.size > 0:
                new_G = S_zono.G
            else:
                new_G = np.array([]).reshape(len(new_c), 0)
            
            return Zonotope(new_c, new_G)

    except Exception as e:
        # Check whether different dimension of ambient space
        if hasattr(S_out, 'dim') and hasattr(S, 'dim'):
            if S_out.dim() != S.dim():
                raise CORAerror('CORA:dimensionMismatch',
                              f'Dimension mismatch: {S_out.dim()} vs {S.dim()}')
        
        # Check for empty sets
        if hasattr(S_out, 'is_empty') and S_out.is_empty():
            return Zonotope.empty(S_out.dim())
        if hasattr(S, 'is_empty') and S.is_empty():
            return Zonotope.empty(S_out.dim())
        
        # Re-raise original error
        raise e
    
    # If we get here, operation is not supported
    raise CORAerror('CORA:noops', f'Operation not supported between {type(S_out)} and {type(S)}')


def _reorder_numeric(Z, S):
    """
    Ensure that numeric is second input argument
    
    Args:
        Z: First operand
        S: Second operand
        
    Returns:
        tuple: (zonotope_operand, other_operand) with zonotope first
    """
    
    # Check for zonotope using both isinstance and class name for robustness
    Z_is_zonotope = isinstance(Z, Zonotope) or (hasattr(Z, '__class__') and Z.__class__.__name__ == 'Zonotope')
    S_is_zonotope = isinstance(S, Zonotope) or (hasattr(S, '__class__') and S.__class__.__name__ == 'Zonotope')
    
    if Z_is_zonotope:
        return Z, S
    elif S_is_zonotope:
        return S, Z
    else:
        raise CORAerror('CORA:wrongInputInConstructor',
                      'At least one operand must be a zonotope') 