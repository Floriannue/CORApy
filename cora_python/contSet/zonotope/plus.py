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
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def plus(Z, S):
    """
    Overloaded '+' operator for Minkowski addition of zonotope with another set or vector
    
    Args:
        Z: zonotope object or numeric
        S: contSet object or numeric
        
    Returns:
        zonotope: Result of Minkowski addition
        
    Raises:
        CORAError: If operation is not supported or dimensions don't match
    """
    from .zonotope import zonotope
    
    # Ensure that numeric is second input argument (reorder if necessary)
    S_out, S = _reorder_numeric(Z, S)
    
    # Call function with lower precedence if applicable
    if hasattr(S, 'precedence') and hasattr(S_out, 'precedence') and S.precedence < S_out.precedence:
        return S + S_out
    
    try:
        # Different cases depending on the class of the second summand
        if isinstance(S, zonotope):
            # Zonotope + Zonotope: see Equation 2.1 in [1]
            new_c = S_out.c + S.c
            if new_c.size == 0:
                return zonotope.empty(S_out.dim())
            
            # Concatenate generator matrices
            if S_out.G.size > 0 and S.G.size > 0:
                new_G = np.hstack([S_out.G, S.G])
            elif S_out.G.size > 0:
                new_G = S_out.G
            elif S.G.size > 0:
                new_G = S.G
            else:
                new_G = np.array([]).reshape(len(new_c), 0)
            
            return zonotope(new_c, new_G)
        
        # Numeric has to be a scalar or a column vector of correct size
        if isinstance(S, (int, float, np.number)):
            # Scalar addition
            new_c = S_out.c + S
            return zonotope(new_c, S_out.G)
        elif isinstance(S, (list, tuple, np.ndarray)):
            S_vec = np.asarray(S).flatten()
            if len(S_vec) == 1:
                # Scalar addition
                new_c = S_out.c + S_vec[0]
                return zonotope(new_c, S_out.G)
            elif len(S_vec) == len(S_out.c):
                # Vector addition
                new_c = S_out.c + S_vec
                return zonotope(new_c, S_out.G)
            else:
                raise CORAError('CORA:wrongInputInConstructor',
                              'Dimension mismatch in addition')
        
        # Handle interval case (if interval class exists)
        if hasattr(S, '__class__') and S.__class__.__name__ == 'interval':
            # Convert interval to zonotope and add
            try:
                from cora_python.contSet.interval.interval import interval
                if isinstance(S, interval):
                    S_zono = zonotope(S)  # This would need interval->zonotope conversion
                    new_c = S_out.c + S_zono.c
                    
                    # Concatenate generator matrices
                    if S_out.G.size > 0 and S_zono.G.size > 0:
                        new_G = np.hstack([S_out.G, S_zono.G])
                    elif S_out.G.size > 0:
                        new_G = S_out.G
                    elif S_zono.G.size > 0:
                        new_G = S_zono.G
                    else:
                        new_G = np.array([]).reshape(len(new_c), 0)
                    
                    return zonotope(new_c, new_G)
            except ImportError:
                pass  # interval class not available

    except Exception as e:
        # Check whether different dimension of ambient space
        if hasattr(S_out, 'dim') and hasattr(S, 'dim'):
            if S_out.dim() != S.dim():
                raise CORAError('CORA:dimensionMismatch',
                              f'Dimension mismatch: {S_out.dim()} vs {S.dim()}')
        
        # Check for empty sets
        if hasattr(S_out, 'is_empty') and S_out.is_empty():
            return zonotope.empty(S_out.dim())
        if hasattr(S, 'is_empty') and S.is_empty():
            return zonotope.empty(S_out.dim())
        
        # Re-raise original error
        raise e
    
    # If we get here, operation is not supported
    raise CORAError('CORA:noops', f'Operation not supported between {type(S_out)} and {type(S)}')


def _reorder_numeric(Z, S):
    """
    Ensure that numeric is second input argument
    
    Args:
        Z: First operand
        S: Second operand
        
    Returns:
        tuple: (zonotope_operand, other_operand) with zonotope first
    """
    from .zonotope import zonotope
    
    if isinstance(Z, zonotope):
        return Z, S
    elif isinstance(S, zonotope):
        return S, Z
    else:
        raise CORAError('CORA:wrongInputInConstructor',
                      'At least one operand must be a zonotope') 