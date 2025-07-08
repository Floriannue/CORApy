"""
and_ - overloads & operator, computes the intersection of two zonotopes

Syntax:
    Z = and_(Z, S)
    Z = and_(Z, S, method)

Inputs:
    Z - zonotope object
    S - contSet object
    method - (optional) algorithm used to compute the intersection
               - 'conZonotope' (default)
               - 'averaging'

Outputs:
    Z - zonotope object enclosing the intersection 

Example: 
    Z1 = Zonotope(np.array([4, 1]), np.array([[2, 2], [2, 0]]))
    Z2 = Zonotope(np.array([3, 3]), np.array([[1, -1, 1], [1, 2, 0]]))
    res = Z1.and_(Z2)

Authors: Matthias Althoff, Niklas Kochdumper, Amr Alanwar (MATLAB)
         Python translation by AI Assistant
Written: 29-June-2009 (MATLAB)
Last update: 02-September-2019 (MATLAB), 28-September-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Any, Optional
from ...g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

from .zonotope import Zonotope


def and_(Z: Zonotope, S: Any, method: Optional[str] = None, *args) -> Zonotope:
    """
    Computes the intersection of two zonotopes
    
    Args:
        Z: First zonotope object
        S: Second contSet object  
        method: Algorithm used to compute the intersection
               - 'conZonotope' (default)
               - 'averaging'
        *args: Additional arguments
        
    Returns:
        Zonotope: Zonotope object enclosing the intersection
        
    Raises:
        CORAerror: If operation not supported between given types
    """
    
    # Default method
    if method is None:
        method = 'conZonotope'
    
    # Call function with lower precedence
    if hasattr(S, 'precedence') and hasattr(Z, 'precedence'):
        if S.precedence < Z.precedence:
            return S.and_(Z, method, *args)
    
    # Quick check: simpler function for intervals
    if Z.representsa_('interval') and S.representsa_('interval'):
        # Convert to intervals and compute exact intersection
        from ..interval import Interval
        Z_interval = Interval(Z)
        S_interval = Interval(S) 
        result_interval = Z_interval.and_(S_interval, 'exact')
        # Convert result interval back to zonotope using the proper method
        return result_interval.zonotope()
    
    if method == 'conZonotope':
        # Convert sets to constrained zonotopes, enclose the resulting
        # intersection by a zonotope
        from ..conZonotope import ConZonotope
        Z_con = ConZonotope(Z)
        S_con = ConZonotope(S)
        result_con = Z_con.and_(S_con, 'exact')
        
        # Check if the result represents an empty set
        if result_con.representsa_('emptySet'):
            return Zonotope.empty(Z.dim())
        
        # Convert result constrained zonotope to zonotope using the zonotope() method
        result_zono = result_con.zonotope()
        
        return result_zono
    
    if method == 'averaging':
        from .private.priv_andAveraging import priv_andAveraging
        result = priv_andAveraging([Z, S], *args)
        
        # Check if the result represents an empty set
        if result.representsa_('emptySet'):
            return Zonotope.empty(Z.dim())
        
        return result
    
    # Throw error for unsupported operations
    raise CORAerror('CORA:noops', str(Z), str(S)) 