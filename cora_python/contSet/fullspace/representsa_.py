"""
representsa_ - checks if a fullspace can also be represented by a
   different set, e.g., a special case

Syntax:
   res = representsa_(fs,type,tol)
   [res,S] = representsa_(fs,type,tol)

Inputs:
   fs - fullspace object
   type - other set representation or 'origin', 'point', 'hyperplane'
   tol - tolerance

Outputs:
   res - true/false
   S - converted set

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/representsa

Authors:       Mark Wetzlinger
Written:       24-July-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def representsa_(fs, type_, tol, *args, **kwargs):
    """
    Checks if a fullspace can also be represented by a
    different set, e.g., a special case
    
    Args:
        fs: fullspace object
        type_: other set representation or 'origin', 'point', 'hyperplane'
        tol: tolerance
        *args: additional arguments
        
    Returns:
        res: true/false
        S: converted set
    """
    # check empty object case
    empty, res, S_conv = fs.representsa_emptyObject(type_)
    
    if empty:
        # Check if we should return converted set based on whether S_conv is requested
        if S_conv is not None:
            return res, S_conv
        else:
            return res
    
    # dimension
    n = fs.dim()
    
    # init second output argument (covering all cases with res = false)
    S = None
    
    if type_ == 'origin':
        # fullspace can never be the origin
        res = False
    
    elif type_ == 'point':
        # fullspace can never be a single point
        res = False
    
    elif type_ == 'capsule':
        # capsules are bounded
        res = False
    
    elif type_ == 'conHyperplane':
        # constrained hyperplanes cannot cover the entire space
        res = False
    
    elif type_ == 'conPolyZono':
        # constrained polynomial zonotopes are bounded
        res = False
    
    elif type_ == 'conZonotope':
        # constrained zonotopes are bounded
        res = False
    
    elif type_ == 'ellipsoid':
        # ellipsoids are bounded
        res = False
    
    elif type_ == 'halfspace':
        # halfspace cannot cover the entire space
        res = False
    
    elif type_ == 'interval':
        # intervals support Inf
        res = True
        if res:  # Always create the converted set if result is True
            # Import here to avoid circular imports
            from cora_python.contSet.interval import Interval
            S = Interval(-np.inf * np.ones(n), np.inf * np.ones(n))
    
    elif type_ == 'levelSet':
        raise CORAerror('CORA:notSupported',
                       f'Comparison of capsule to {type_} not supported.')
    
    elif type_ == 'polytope':
        # polytopes cannot cover the entire space
        res = False
    
    elif type_ == 'polyZonotope':
        # polynomial zonotopes are bounded
        res = False
    
    elif type_ == 'probZonotope':
        res = False
    
    elif type_ == 'zonoBundle':
        # zonotope bundles are bounded
        res = False
    
    elif type_ == 'zonotope':
        # zonotopes are bounded
        res = False
    
    elif type_ == 'hyperplane':
        # hyperplanes cannot cover the entire space
        res = False
    
    elif type_ == 'parallelotope':
        # parallelotopes are bounded
        res = False
    
    elif type_ == 'convexSet':
        res = True
    
    elif type_ == 'emptySet':
        res = False
    
    elif type_ == 'fullspace':
        # obviously true
        res = True
        S = fs
    
    # Always return the result, and include converted set if it exists
    if S is not None:
        return res, S
    else:
        return res

# ------------------------------ END OF CODE ------------------------------ 