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
from typing import TYPE_CHECKING, Union, Tuple

from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def contains(S1: 'ContSet', S2: Union['ContSet', np.ndarray], method='exact', tol=None, maxEval=None, *, return_cert=False, return_scaling=False):
    """
    Determines if a set contains another set or a point.
    
    Args:
        S1: contSet object
        S2: contSet object or numeric array
        method: method for computation (default: 'exact')
        tol: tolerance (default: 100*eps)
        maxEval: maximal number of iterations (default: depends on S2)
        return_cert: whether to return certification information
        return_scaling: whether to return scaling factor
        
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
    
    # Check input arguments using inputArgsCheck (following MATLAB exactly)
    inputArgsCheck([
        [S1, 'att', 'contSet'],
        [S2, 'att', ['contSet', 'taylm', 'numeric']],
        [method, 'str', ['exact', 'exact:venum', 'exact:polymax',
                        'exact:zonotope', 'exact:polytope',
                        'approx', 'approx:st', 'approx:stDual',
                        'opt',
                        'sampling', 'sampling:primal', 'sampling:dual']],
        [tol, 'att', 'numeric', ['scalar', 'nonnegative', 'nonnan']],
        [maxEval, 'att', 'numeric', ['scalar', 'nonnegative', 'nonnan']]
    ])
    
    # --- Empty Set Handling ---
    # Outer body is empty - check this first
    if hasattr(S1, 'isemptyobject') and S1.isemptyobject():
        res, cert, scaling = False, True, np.inf
        if return_scaling:
            return res, cert, scaling
        elif return_cert:
            return res, cert
        return res
    
    # Inner-body is empty numeric array or empty contSet
    if (isinstance(S2, np.ndarray) and S2.size == 0) or \
       (hasattr(S2, 'isemptyobject') and S2.isemptyobject()):
        res, cert, scaling = True, True, 0
        if return_scaling:
            return res, cert, scaling
        elif return_cert:
            return res, cert
        return res

    try:
        # Check dimension mismatch
        equal_dim_check(S1, S2)
        
        # Call subclass method with appropriate toggles
        cert_toggle = return_cert or return_scaling  # Only compute cert if needed
        scaling_toggle = return_scaling  # Only compute scaling if needed
        res, cert, scaling = S1.contains_(S2, method, tol, maxEval, cert_toggle, scaling_toggle)
        
        # Return based on what's requested
        if return_scaling:
            # Convert arrays to scalars for single points
            if isinstance(res, np.ndarray) and res.size == 1:
                res = res.item()
            if isinstance(cert, np.ndarray) and cert.size == 1:
                cert = cert.item()
            if isinstance(scaling, np.ndarray) and scaling.size == 1:
                scaling = scaling.item()
            return res, cert, scaling
        elif return_cert:
            # Convert arrays to scalars for single points
            if isinstance(res, np.ndarray) and res.size == 1:
                res = res.item()
            if isinstance(cert, np.ndarray) and cert.size == 1:
                cert = cert.item()
            return res, cert
        else:
            # For single boolean result, ensure we return a scalar
            if isinstance(res, np.ndarray) and res.size == 1:
                return res.item()
            elif isinstance(res, np.ndarray) and res.ndim == 1 and len(res) > 0:
                # For multiple identical values (like [True, True]), check if all are the same
                if np.all(res == res[0]):
                    return res[0].item() if hasattr(res[0], 'item') else res[0]
                else:
                    return res
            else:
                return res
            
    except Exception as e:
        # Re-raise if not an empty set case
        raise e 