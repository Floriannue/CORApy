"""
contains - determines if a set contains another set or a point

Syntax:
    res = contains(S1,S2)
    res = contains(S1,S2,method)
    res = contains(S1,S2,method,tol)
    res = contains(S1,S2,method,tol,maxEval)
    res, cert = contains(___)
    res, cert, scaling = contains(___)

Inputs:
    S1 - contSet object
    S2 - contSet object, numeric array
    method - method for computation ('exact' or 'approx', or any method
           name specific to a certain set representation; see the
           documentation of the corresponding contains_ function, e.g.,
           zonotope/contains_ )
    tol - tolerance
    maxEval - maximal number of iterations for optimization-based methods.
            See the corresponding contains_ (e.g., zonotope/contains_)
            function for more details

Outputs:
    res - true if S2 is contained in S1, false if S2 is not contained
          S1, or if the containment could not be certified
    cert - (optional) returns true iff the result of res could be
            verified. For example, if res=false and cert=true, S2 is
            guaranteed to not be contained in S1, whereas if res=false and
            cert=false, nothing can be deduced (S2 could still be
            contained in S1).
            If res=true, then cert=true.
            Note that computing this certification may marginally increase
            the runtime.
    scaling - (optional) assuming S1 has a center, returns the smallest
            number 'scaling', such that
            scaling*(S1 - S1.center) + S1.center contains S2.
            For methods other than 'exact', this may be a lower or upper
            bound on the exact number. See the corresponding contains_
            (e.g., zonotope/contains_) for more details.
            Note that computing this scaling factor may significantly
            increase the runtime. This scaling can be used in
            contSet/enlarge.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: - 

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       18-August-2022
Last update:   23-November-2022 (MW, add classname as input argument)
Last revision: 27-March-2023 (MW, restructure relation to subclass)
"""

import numpy as np
from .isemptyobject import isemptyobject
from .contains_ import contains_

def contains(S1, S2, method='exact', tol=None, maxEval=None):
    """
    Determines if a set contains another set or a point.
    
    Args:
        S1: contSet object
        S2: contSet object or numeric array
        method: method for computation (default: 'exact')
        tol: tolerance (default: 100*eps)
        maxEval: maximal number of iterations (default: depends on S2)
        
    Returns:
        bool or tuple: containment result, optionally with certification and scaling
    """
    # Default values
    if tol is None:
        tol = 100 * np.finfo(float).eps
    
    if maxEval is None:
        # Default value for maxEval depends on in-body zonotope
        if hasattr(S2, '__class__') and S2.__class__.__name__ == 'Zonotope':
            if hasattr(S2, 'generators'):
                gen_shape = S2.generators().shape if hasattr(S2.generators(), 'shape') else (0, 0)
                maxEval = max(500, 200 * gen_shape[1])
            else:
                maxEval = 500
        else:
            maxEval = 200
    
    # Validate method
    valid_methods = [
        'exact', 'exact:venum', 'exact:polymax',
        'exact:zonotope', 'exact:polytope',
        'approx', 'approx:st', 'approx:stDual',
        'opt',
        'sampling', 'sampling:primal', 'sampling:dual'
    ]
    
    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method}")
    
    # Validate input arguments
    if not isinstance(tol, (int, float)) or tol < 0 or np.isnan(tol):
        raise ValueError("Tolerance must be a non-negative number")
    
    if not isinstance(maxEval, (int, float)) or maxEval < 0 or np.isnan(maxEval):
        raise ValueError("maxEval must be a non-negative number")
    
    try:
        # Check dimension mismatch (would need to implement equalDimCheck)
        # For now, skip this check
        
        # Call subclass method
        from .contains_ import contains_
        result = contains_(S1, S2, method, tol, maxEval, cert_toggle=True, scaling_toggle=True)
        
        # Handle return values based on what's requested
        if isinstance(result, tuple):
            return result
        else:
            return result
            
    except Exception as e:
        # Handle empty set cases
        
        # Inner-body is empty numeric array or empty contSet
        if (isinstance(S2, np.ndarray) and S2.size == 0) or \
           (hasattr(S2, '__class__') and hasattr(S2, 'isemptyobject') and isemptyobject(S2)):
            return True, True, 0
        
        # Outer body is empty
        if hasattr(S1, 'isemptyobject') and isemptyobject(S1):
            return False, True, np.inf
        
        # Re-raise if not an empty set case
        raise e 