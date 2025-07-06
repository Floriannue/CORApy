"""
representsa_ - checks if a polytope represents a specific type

Syntax:
    res = representsa_(P, type, tol)

Inputs:
    P - polytope object
    type - string specifying the type ('emptySet', 'fullspace', 'origin', etc.)
    tol - tolerance for checks

Outputs:
    res - true/false

Authors:       Mark Wetzlinger
Written:       19-July-2023
Last update:   ---
Last revision: ---
"""

from typing import Tuple, Union
import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from .private.priv_representsa_emptySet import priv_representsa_emptySet


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
        # The 'emptySet' property is computed during construction
        res = p.emptySet
            
    elif set_type == 'point':
        if p._isVRep:
            V = p._V
            if V is None or V.size == 0 or V.shape[1] == 0:
                res = False  # Empty set is not a point
            elif V.shape[1] == 1:
                res = True  # Single vertex
            else:
                # Check if all vertices are the same
                res = np.all(withinTol(V, V[:, [0]], tol))
        else:
            # For H-representation, have to trigger vertex computation
            from .vertices_ import vertices_
            V = vertices_(p)
            if V is None or V.size == 0 or V.shape[1] == 0:
                res = False
            elif V.shape[1] == 1:
                res = True
            else:
                res = np.all(withinTol(V, V[:, [0]], tol))
                
    elif set_type == 'fullspace':
        # MATLAB behavior: check H-rep constraints if available, otherwise check V-rep for 1D infinite vertices
        if p._isHRep:
            # All constraints must be trivially fulfilled: A*x <= b with A=0, b>=0, Ae=0, be=0
            A, b, Ae, be = p._A, p._b, p._Ae, p._be
            res = (A is None or A.size == 0 or np.all(withinTol(A, 0, tol))) and \
                  (b is None or b.size == 0 or np.all(b >= -tol)) and \
                  (Ae is None or Ae.size == 0 or np.all(withinTol(Ae, 0, tol))) and \
                  (be is None or be.size == 0 or np.all(withinTol(be, 0, tol)))
        elif p._isVRep and n == 1:
            # For 1D V-representation, check for infinite vertices
            V = p._V
            res = np.any(V == -np.inf) and np.any(V == np.inf)
        else:
            # For nD V-representation, cannot be fullspace (finite vertices)
            res = False
            
    elif set_type == 'origin':
        if p._isVRep:
            V = p._V
            if V is None or V.size == 0 or V.shape[1] == 0:
                res = False  # Empty set is not origin
            else:
                # Check if all vertices are at origin
                res = np.all(withinTol(V, 0, tol))
        else:
            # For H-representation, check if origin satisfies all constraints
            A, b, Ae, be = p._A, p._b, p._Ae, p._be
            # Check Ax <= b (0 <= b) and Aex = be (0 = be)
            res = (A is None or b is None or np.all(b >= -tol)) and \
                  (Ae is None or be is None or np.all(withinTol(be, 0, tol)))

    elif set_type == 'conHyperplane':
        # Constrained hyperplane: exactly one equality constraint
        if p._Ae is not None and p._be is not None:
            res = len(p._be.flatten()) == 1
        else:
            res = False

    if 'return_set' in kwargs and kwargs['return_set']:
        if res:
            if set_type == 'point':
                # Return the actual point coordinates, not the polytope object
                if p._isVRep:
                    V = p._V
                    if V is not None and V.size > 0 and V.shape[1] > 0:
                        p_conv = V[:, [0]]  # Return first vertex as point coordinates
                    else:
                        p_conv = None
                else:
                    # For H-representation, get vertices first
                    from .vertices_ import vertices_
                    V = vertices_(p)
                    if V is not None and V.size > 0 and V.shape[1] > 0:
                        p_conv = V[:, [0]]  # Return first vertex as point coordinates
                    else:
                        p_conv = None
            else:
                p_conv = p
        return res, p_conv
    else:
        return res 