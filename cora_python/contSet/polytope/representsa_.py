from typing import Tuple, Union
import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

def representsa_(p: 'Polytope', set_type: str, tol: float = 1e-9, **kwargs) -> Union[bool, Tuple[bool, 'Polytope']]:
    """
    Checks if a polytope can be represented by another set type.
    
    Args:
        p: Polytope object
        set_type: String representing the target set type
        tol: Tolerance for numerical comparisons
        **kwargs: Additional arguments
        
    Returns:
        bool or tuple: True/False or (True/False, converted_polytope)
    """
    res = False
    p_conv = None
    n = p.dim()

    if set_type == 'emptySet':
        # Check if emptiness already known
        if hasattr(p, '_empty_set_cached') and p._empty_set_cached is not None:
            res = p._empty_set_cached
        elif p._has_v_rep:
            # For V-representation, just check if no vertices
            res = p._V.size == 0 or p._V.shape[1] == 0
        else:
            # For H-representation, need to call the private function
            from .private.priv_representsa_emptySet import priv_representsa_emptySet
            res = priv_representsa_emptySet(p.A, p.b, p.Ae, p.be, n, tol)
        
        # Cache the result
        p._empty_set_cached = res
            
    elif set_type == 'point':
        if p._has_v_rep:
            V = p._V
            if V.size == 0 or V.shape[1] == 0:
                res = False  # Empty set is not a point
            elif V.shape[1] == 1:
                res = True  # Single vertex
            else:
                # Check if all vertices are the same
                res = np.all(withinTol(V, V[:, [0]], tol))
        else:
            # For H-representation, would need more complex check
            # For now, delegate to vertex computation
            V = p.V  # This will trigger V-rep computation
            if V.size == 0 or V.shape[1] == 0:
                res = False
            elif V.shape[1] == 1:
                res = True
            else:
                res = np.all(withinTol(V, V[:, [0]], tol))
                
    elif set_type == 'fullspace':
        # MATLAB behavior: check H-rep constraints if available, otherwise check V-rep for 1D infinite vertices
        if p._has_h_rep:
            # All constraints must be trivially fulfilled: A*x <= b with A=0, b>=0, Ae=0, be=0
            A, b, Ae, be = p.A, p.b, p.Ae, p.be
            res = (np.all(withinTol(A, 0, tol)) and
                   np.all(b >= 0) or np.all(withinTol(b, 0, tol))) and \
                  (Ae is None or Ae.size == 0 or np.all(withinTol(Ae, 0, tol))) and \
                  (be is None or be.size == 0 or np.all(withinTol(be, 0, tol)))
        elif p._has_v_rep and n == 1:
            # For 1D V-representation, check for infinite vertices
            V = p._V
            res = np.any(V == -np.inf) and np.any(V == np.inf)
        else:
            # For nD V-representation, cannot be fullspace (finite vertices)
            res = False
            
    elif set_type == 'origin':
        if p._has_v_rep:
            V = p._V
            if V.size == 0 or V.shape[1] == 0:
                res = False  # Empty set is not origin
            else:
                # Check if all vertices are at origin
                res = np.all(withinTol(V, 0, tol))
        else:
            # For H-representation, would need to check if origin satisfies all constraints
            # For now, delegate to vertex computation
            V = p.V  # This will trigger V-rep computation
            if V.size == 0 or V.shape[1] == 0:
                res = False
            else:
                res = np.all(withinTol(V, 0, tol))

    if 'return_set' in kwargs and kwargs['return_set']:
        if res:
            p_conv = p
        return res, p_conv
    else:
        return res 