"""
representsa_ - checks if a zonotope can also be represented by a
   different set, e.g., a special case

Syntax:
   res = representsa_(Z, type, tol)
   [res, S] = representsa_(Z, type, tol)

Inputs:
   Z - zonotope object
   type - other set representation or 'origin', 'point', 'hyperplane'
   tol - tolerance

Outputs:
   res - true/false
   S - converted set

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/representsa

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 19-July-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Union, Optional, Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def representsa_(Z, set_type: str, tol: float = 1e-12, method: str = 'linearize', iter_val: int = 1, splits: int = 0, **kwargs):
    """
    Checks if a zonotope can also be represented by a different set type
    
    Args:
        Z: zonotope object or numpy array
        set_type: string indicating the target set type
        tol: tolerance (default: 1e-12)
        **kwargs: additional arguments
        
    Returns:
        bool: Whether Z can be represented by set_type (if called with single output)
        tuple: (bool, converted_set) if called expecting two outputs
    """
    # Import here to avoid circular imports
    from cora_python.contSet.interval import Interval
    from cora_python.contSet.capsule import Capsule
    
    # Handle numpy arrays - they cannot represent empty sets
    if isinstance(Z, np.ndarray):
        if set_type == 'emptySet':
            return False
        # For other types, would need conversion to zonotope first
        return False
    
    # Check if we need to return the converted set
    return_set = 'return_set' in kwargs and kwargs['return_set']
    
    # Check empty object case
    # TODO: Implement representsa_emptyObject when needed
    
    # Dimension
    n = Z.dim()
    
    # Init second output argument (covering all cases with res = false)
    S = None
    
    # Delete zero-length generators and compact the zonotope
    Z_compact = Z.compact_('zeros', np.finfo(float).eps)
    
    # Number of generators
    nrGens = Z_compact.G.shape[1] if Z_compact.G.size > 0 else 0
    
    if set_type == 'origin':
        c = Z_compact.c
        if c is None or c.size == 0:
            res = False
        elif nrGens == 0:
            # No generators, just check if center is close to origin
            res = np.allclose(c, 0, atol=tol)
        else:
            # Check if the interval norm is within tolerance
            try:
                I = Z_compact.interval()
                res = I.norm() <= tol
            except:
                # Fallback: use more conservative check
                res = np.linalg.norm(c) <= tol and np.linalg.norm(Z_compact.G, 'fro') <= tol
        
        if return_set and res:
            S = np.zeros((n, 1))
    
    elif set_type == 'point':
        if nrGens == 0:
            res = True
        else:
            # Check if all generators are within tolerance
            try:
                Z_centered = Z_compact - Z_compact.c
                I = Z_centered.interval()
                res = I.norm() <= tol
            except:
                # Fallback: check generator matrix norm
                res = np.linalg.norm(Z_compact.G, 'fro') <= tol
        
        if return_set and res:
            S = Z_compact.c
    
    elif set_type == 'capsule':
        # True if only one or less generators
        res = n == 1 or nrGens <= 1
        if return_set and res:
            S = Capsule(Z_compact.c, Z_compact.G, 0)
    
    elif set_type == 'interval':
        # Check if zonotope is axis-aligned (generators are axis-aligned)
        if nrGens == 0:
            res = True
        else:
            # Check if all generators are axis-aligned
            G = Z_compact.G
            res = True
            for i in range(G.shape[1]):
                gen = G[:, i]
                # Generator is axis-aligned if it has only one non-zero entry
                non_zero_count = np.sum(np.abs(gen) > tol)
                if non_zero_count > 1:
                    res = False
                    break
        if return_set:
            if res:
                S = Z_compact.interval()
            else:
                S = None
    
    elif set_type == 'zonotope':
        # Obviously true
        res = True
        if return_set:
            S = Z_compact
    
    elif set_type == 'emptySet':
        # Check if zonotope represents empty set
        res = Z.isemptyobject()
    
    elif set_type == 'fullspace':
        # Zonotopes are bounded, so never fullspace
        res = False

    elif set_type == 'parallelotope':
        # Check if zonotope is a parallelotope (n generators in n dimensions)
        if nrGens == 0:
            res = False  # Point is not a parallelotope
        elif n == 1 and nrGens > 0:
            res = True  # 1D zonotopes with generators are parallelotopes
        else:
            # Check if we have exactly n linearly independent generators
            if nrGens < n:
                res = False
            elif nrGens == n:
                # Check if generators are linearly independent
                try:
                    rank = np.linalg.matrix_rank(Z_compact.G)
                    res = rank == n
                except:
                    res = False
            else:
                res = False  # Too many generators
        
        if return_set and res:
            S = Z_compact

    elif set_type == 'conZonotope':
        # Every zonotope is a constrained zonotope
        res = True
        if return_set:
            # For now, return the zonotope itself (would need conZonotope class)
            S = Z_compact

    elif set_type == 'polyZonotope':
        # Every zonotope is a polynomial zonotope
        res = True
        if return_set:
            # For now, return the zonotope itself (would need polyZonotope class)
            S = Z_compact

    else:
        raise CORAerror("CORA:notSupported",
                       f"Comparison of zonotope to {set_type} not supported.")
    
    if return_set:
        return res, S
    else:
        return res 