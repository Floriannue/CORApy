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

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.optimize import linprog
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.contSet.polytope.private.priv_emptySet import priv_emptySet
from cora_python.contSet.polytope.private.priv_compact_zeros import priv_compact_zeros
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_toEquality import priv_compact_toEquality
from typing import TYPE_CHECKING, Tuple, Union, Optional

if TYPE_CHECKING:
    from .polytope import Polytope


def representsa_(p: 'Polytope', set_type: str, tol: float = 1e-9, **kwargs) -> Union[bool, Tuple[bool, np.ndarray]]:
    """
    Checks if a polytope can be represented by another set type.
    
    Args:
        p: Polytope object
        set_type: String representing the target set type
        tol: Tolerance for numerical comparisons
        **kwargs: Additional arguments
        
    Returns:
        bool or tuple: True/False or (True/False, converted_object)
    """
    
    def _return_result(res_val, obj_val=None):
        """Helper to handle return logic based on return_set parameter"""
        if 'return_set' in kwargs and kwargs['return_set']:
            return res_val, obj_val
        return res_val
    
    res = False
    return_obj = None
    n = p.dim()

    # Handle case where p.Ae or p.be might be None
    # Ensure they are empty numpy arrays if not initialized to prevent AttributeError
    if p.Ae is None:
        p.Ae = np.array([[]]).reshape(0, n)
    if p.be is None:
        p.be = np.array([[]]).reshape(0, 1)

    # Check for empty object case using isemptyobject like MATLAB
    empty_obj = p.isemptyobject()
    if empty_obj:
        if set_type == 'emptySet':
            res = True
            # MATLAB: P.emptySet.val = res; (already known to be true)
            # save properties, now that P is known to be the empty set
            if res:
                p._bounded_val = True      # MATLAB: P.bounded.val = true;
                p._fullDim_val = False     # MATLAB: P.fullDim.val = false;
                # Note: MATLAB also sets P.V_.val = zeros(n,0), P.isVRep.val = true, P.minVRep.val = true
                # But we handle representation flags differently in Python
            if 'return_set' in kwargs and kwargs['return_set']:
                from cora_python.contSet.emptySet.emptySet import EmptySet
                return_obj = EmptySet(n)
            return _return_result(res, return_obj)
        elif set_type == 'fullspace':
            # If the polytope is empty, it cannot be a fullspace
            return _return_result(False, None)
        else:
            # Empty polytope cannot represent other types
            return _return_result(False, None)
    elif set_type == 'emptySet': 
        return _return_result(False, None)
    
    # If it's a fullspace, emptySet would be false. Check fullspace explicitly.
    elif set_type == 'fullspace': # Moved to an elif to correctly return False if not fullspace
        # MATLAB logic for fullspace check:
        # P.isHRep.val && all(all(withinTol(P.A_.val,0,tol))) && all(P.b_.val > 0 | withinTol(P.b_.val,0,tol)) && ...
        # all(all(withinTol(P.Ae_.val,0,tol))) && all(withinTol(P.be_.val,0,tol))) || ...
        # (P.isVRep.val && (n == 1 && any(P.V_.val == -Inf) && any(P.V_.val == Inf)));
        
        # Check H-representation
        hrep_fullspace = (p.isHRep and \
                          np.all(withinTol(p.A, 0, tol)) and \
                          np.all((p.b > 0) | withinTol(p.b, 0, tol)))
        
        # Check equality constraints for H-representation
        # Ensure Ae and be are also trivially fulfilled (all zeros) for fullspace
        if p.Ae.size > 0:
            hrep_fullspace = hrep_fullspace and \
                             np.all(withinTol(p.Ae, 0, tol)) and \
                             np.all(withinTol(p.be, 0, tol))
        
        # Check V-representation (1D case with -Inf and Inf)
        vrep_fullspace = False
        if p.isVRep and n == 1:
            # Need to get vertices to check for +/-Inf
            # This would require calling p.vertices_() which might cause a circular dependency or
            # be problematic if vertices_ relies on representsa_
            # For now, we'll assume isVRep is only true if vertices have been computed and stored
            # and that P.V is accessible and reflects the V-representation.
            # If P.V is not directly available/cached, we might need a more robust way
            # or rely purely on H-rep for now.
            # Given that isFullDim and isBounded are properties that may trigger computation,
            # this might be the source of issues.
            # Let's try to directly replicate the MATLAB condition first:
            # any(P.V_.val == -Inf) && any(P.V_.val == Inf)
            try:
                # Access P.V property (which should trigger computation if not computed)
                vertices = p.V
                vrep_fullspace = (n == 1 and np.any(np.isinf(vertices) & (vertices < 0)) and \
                                  np.any(np.isinf(vertices) & (vertices > 0)))
            except Exception: # Catch any error during P.V access, e.g., if V is not computed
                vrep_fullspace = False
        
        res = hrep_fullspace or vrep_fullspace
        
        # fullspaces are always unbounded, non-empty and full-dimensional
        if res:
            p._bounded_val = False     # MATLAB: P.bounded.val = false;
            p._emptySet_val = False    # MATLAB: P.emptySet.val = false;
            p._fullDim_val = True      # MATLAB: P.fullDim.val = true;
        
        if res and 'return_set' in kwargs and kwargs['return_set']:
            from cora_python.contSet.fullspace.fullspace import fullspace as fullspace_func
            return_obj = fullspace_func(n) # Return an actual fullspace object
        return _return_result(res, return_obj)

    # Initialize return_obj (for cases where res is False by default)
    if 'return_set' in kwargs and kwargs['return_set']:
        return_obj = None # Default to None if not converting
    
    # Logic for other types
    if set_type == 'origin':

        # Quick check: is origin contained?
        # Use P.contains_ for a robust check
        if p.contains_(np.zeros((n, 1)), 'exact', tol, certToggle=False, scalingToggle=False)[0]:
            # Definitely not empty if the origin is contained
    
            

            # Check if only origin contained by checking the norm of its interval hull
            # MATLAB: norm_(interval(P),2) <= tol
            P_interval = p.interval()
            # Simple check: max absolute value of interval bounds is near zero.
            if np.all(np.abs(P_interval.inf) <= tol) and np.all(np.abs(P_interval.sup) <= tol):
                 res = True
                 # Set is degenerate if it's only the origin
         
                 
                 if 'return_set' in kwargs and kwargs['return_set']:
                     return_obj = np.zeros((n, 1)) # Return as numpy array for origin point
        return _return_result(res, return_obj)

    elif set_type == 'point':
        if n == 1:
            V = p.vertices_()
            res = V.shape[1] == 1 # True if only one vertex
        else:
            # MATLAB: [fulldim,subspace] = isFullDim(P); res = ~fulldim && isempty(subspace);
            fulldim, subspace = p.isFullDim()
            res = not fulldim and (subspace is None or subspace.size == 0)
        
        # Set is degenerate if it's only a single point
        if res:
            p._fullDim_val = False     # MATLAB: P.fullDim.val = false;
        
        # Only compute center if return_set is requested AND res is True (like MATLAB nargout == 2 && res)
        if 'return_set' in kwargs and kwargs['return_set'] and res:
            # MATLAB: if P.isVRep.val, S = center(P, 'avg'); else S = center(P); end
            if hasattr(p, 'isVRep') and p.isVRep:
                return_obj = p.center('avg')
            else:
                return_obj = p.center()
        
        return _return_result(res, return_obj)

    elif set_type == 'capsule':
        # True if 1D and bounded (note: also true if polytope is a bounded line)
        # MATLAB uses isBounded(P) which might trigger computation
        res = (n == 1 and p.bounded and not p.emptySet) # Use p.bounded property
        if 'return_set' in kwargs and kwargs['return_set']:
            if res:
                # If it represents a capsule and return_set is true, construct and return it
                # For now, we return None as capsule conversion is not fully supported
                return_obj = None
            else:
                return_obj = None
            raise CORAerror('CORA:notSupported',
                            f'Conversion from polytope to {set_type} not supported.')
        return _return_result(res, return_obj)

    elif set_type == 'conHyperplane':

        # MATLAB uses aux_isConHyperplane, which computes a minimal representation
        # constrained hyperplane only makes sense if it is (n-1)-dimensional
        # (or empty) so that we can use it for guard intersection: this occurs
        # if and only if there is exactly one equality constraints

        # Compute truly minimal representation and try again
        A, b, Ae, be, empty = priv_compact_zeros(p.A, p.b, p.Ae, p.be, tol)
        if empty:
            res = False
            if 'return_set' in kwargs and kwargs['return_set']:# Added this block to ensure tuple return
                return res, None
            return res, None # Ensure tuple return here

        A, b, Ae, be = priv_normalizeConstraints(A, b, Ae, be, 'A')
        A, b, Ae, be = priv_compact_toEquality(A, b, Ae, be, tol)

        res = Ae.shape[0] == 1 and A.size == 0 and be.size > 0 # Exactly one equality, no inequalities, non-empty be

        if res: # If it's a constrained hyperplane
            # hyperplanes are unbounded and non-empty
    
            
    
            
            # Full dimensional in its subspace (n-1 dim), but not n-dim
    
            
            # Note: MATLAB returns converted conHyperplane object, but since we don't have 
            # conHyperplane implemented yet, we return None for now
            # TODO: Implement conHyperplane conversion when the class is available
            return_obj = None
        return _return_result(res, return_obj)

    elif set_type == 'conPolyZono':
        res = p.bounded
        if 'return_set' in kwargs and kwargs['return_set'] and res:
            from cora_python.contSet.conPolyZono.conPolyZono import conPolyZono # Placeholder
            return_obj = conPolyZono(p) # Assumes conversion constructor
        return _return_result(res, return_obj)

    elif set_type == 'conZonotope':
        res = p.bounded
        if 'return_set' in kwargs and kwargs['return_set'] and res:
            from cora_python.contSet.conZonotope.conZonotope import conZonotope # Placeholder
            return_obj = conZonotope(p) # Assumes conversion constructor
        return _return_result(res, return_obj)

    elif set_type == 'ellipsoid':
        # only an ellipsoid if 1D and bounded or a single point
        res = (n == 1 and p.bounded and not p.emptySet) or p.representsa_('point', tol) # Use p.bounded
        if 'return_set' in kwargs and kwargs['return_set'] and res:
            from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid # Placeholder
            return_obj = Ellipsoid(p) # Assumes conversion constructor
        return _return_result(res, return_obj)

    elif set_type == 'halfspace':
        # MATLAB uses aux_isHalfspace
        # At least one equality constraint -> cannot be a halfspace
        if p.Ae.size > 0: # Use p.Ae directly
            res = False
        else:
            # Remove all-zero constraints and normalize
            A, b, _, _, empty = priv_compact_zeros(p.A, p.b, p.Ae, p.be, tol)
            if empty:
                res = False # Empty set is not a halfspace
            else:
                # MATLAB: A,b,Ae,be = priv_normalizeConstraints(A,b,[],[],'A');
                A, b, _, _ = priv_normalizeConstraints(A, b, np.array([[]]).reshape(0,n), np.array([[]]).reshape(0,1), 'A') # Normalize
                # Check if all inequality constraints are aligned (dot product of A with A.T is identity)
                if A.shape[0] > 0 and A.shape[1] > 0:
                    dot_product = A @ A.T
                    # A halfspace has only one distinct constraint, so A.shape[0] should be 1 (after reduction)
                    res = dot_product.shape == (1,1) and withinTol(dot_product[0,0], 1, tol)
                else:
                    res = False # No constraints -> fullspace, not halfspace

        # Note: MATLAB doesn't convert to halfspace object, just returns boolean
        return _return_result(res, return_obj)

    elif set_type == 'interval':
        # MATLAB uses aux_isInterval
        n_curr = p.dim()
        lb = np.full((n_curr, 1), -np.inf)
        ub = np.full((n_curr, 1), np.inf)

        # Temporary import for Interval class, to be replaced by dynamic import if needed
        from cora_python.contSet.interval.interval import Interval

        all_constraints_A = np.vstack([p.A, p.Ae, -p.Ae])
        all_constraints_b = np.vstack([p.b, p.be, -p.be])
        nrCon = all_constraints_A.shape[0]
        checkRedundancy_mask = np.zeros(nrCon, dtype=bool)

        res = True # Assume true until proven otherwise
        
        for c_idx in range(nrCon):
            constraint = all_constraints_A[c_idx, :].reshape(-1, 1)
            offset = all_constraints_b[c_idx, 0]

            if not np.any(withinTol(constraint, 0, tol)): # If not all-zero constraint
                # Find non-zero index (axis-aligned)
                non_zero_indices = np.where(np.abs(constraint.flatten()) > tol)[0]
                if len(non_zero_indices) == 1:
                    idx = non_zero_indices[0]
                    constraint_val = constraint[idx, 0]
                    if constraint_val > 0:
                        ub[idx, 0] = np.min([offset / constraint_val, ub[idx, 0]])
                    else:
                        lb[idx, 0] = np.max([offset / constraint_val, lb[idx, 0]])
                else:
                    # Non-axis-aligned constraint, check redundancy later
                    checkRedundancy_mask[c_idx] = True
            elif offset < -tol: # All-zero constraint with negative offset -> empty set
                res = False
                break

        if not res: # If already determined as empty
            if 'return_set' in kwargs and kwargs['return_set']:
                return_obj = Interval.empty(n_curr)
            return _return_result(res, return_obj)
        I = Interval(lb, ub)
        
        # Go over all non-axis-aligned constraints and check redundancy
        if np.any(checkRedundancy_mask):
            for c_idx in range(nrCon):
                if checkRedundancy_mask[c_idx]:
                    constraint = all_constraints_A[c_idx, :].reshape(-1, 1)
                    offset = all_constraints_b[c_idx, 0]
                    
                    val_upper = I.supportFunc_(constraint, 'upper')
                    if val_upper <= offset + tol: # Constraint is redundant (or satisfied within tolerance)
                        # Constraint is redundant
                        pass
                    else:
                        # Constraint cuts through the interval. Check lower bound of support function
                        val_lower = I.supportFunc_(constraint, 'lower')
                        if val_lower > offset - tol: # Interval is on one side, meaning empty with this constraint
                            res = False # Empty
                            break
                        else:
                            # Constraint cuts through the interval, so P is not an interval
                            # Fallback to checking if P is an empty set
                            if p.emptySet:
                                res = True # Yes, it's an interval, but it's empty (consistent with MATLAB)
                                if 'return_set' in kwargs and kwargs['return_set']:
                                     return_obj = Interval.empty(n_curr)
                            else:
                                res = False # Not an interval
                            break
        
        if res and 'return_set' in kwargs and kwargs['return_set']:
            return_obj = I
        else:
            return_obj = None # Explicitly set to None if return_set is false or res is False
        return _return_result(res, return_obj)

    elif set_type == 'levelSet':
        res = True
        if 'return_set' in kwargs and kwargs['return_set']:
            from cora_python.contSet.levelSet.levelSet import levelSet as levelSet_func # Placeholder
            return_obj = levelSet_func(p) # Assumes conversion constructor
        return _return_result(res, return_obj)

    elif set_type == 'polytope':
        res = True
        if 'return_set' in kwargs and kwargs['return_set']:
            return_obj = p
        return _return_result(res, return_obj)

    elif set_type == 'polyZonotope':
        res = p.bounded
        if 'return_set' in kwargs and kwargs['return_set'] and res:
            from cora_python.contSet.polyZonotope.polyZonotope import polyZonotope # Placeholder
            return_obj = polyZonotope(p) # Assumes conversion constructor
        return _return_result(res, return_obj)

    elif set_type == 'probZonotope':
        res = False # Not supported / always false in MATLAB
        return _return_result(res, None)

    elif set_type == 'zonoBundle':
        res = p.bounded
        if 'return_set' in kwargs and kwargs['return_set'] and res:
            from cora_python.contSet.zonoBundle.zonoBundle import zonoBundle # Placeholder
            return_obj = zonoBundle(p) # Assumes conversion constructor
        return _return_result(res, return_obj)

    elif set_type == 'zonotope':
        raise CORAerror('CORA:notSupported',
                        f'Comparison of polytope to {set_type} not supported.')

    elif set_type == 'hyperplane':
        # Hyperplane: no inequality constraints, exactly one equality constraint
        # Use p.A and p.Ae directly
        res = p.A.size == 0 and p.Ae.shape[0] == 1
        return _return_result(res, return_obj)

    elif set_type == 'parallelotope':
        raise CORAerror('CORA:notSupported',
                        f'Comparison of polytope to {set_type} not supported.')

    elif set_type == 'convexSet':
        res = True
        if 'return_set' in kwargs and kwargs['return_set']: # If return_set is true, return None
            return res, None
        return res, None # Ensure tuple return here

    # Note: 'emptySet' and 'fullspace' are handled at the beginning due to their interaction
    # with P.emptySet and P.fullDim properties. This deviates from MATLAB's switch structure
    # but is more efficient as these properties are computed/cached.

    else:
        raise CORAerror('CORA:wrongValue', 'second', f"Unknown set type '{set_type}'.")

    # This final return is now redundant as all branches should return already
    # if 'return_set' in kwargs and kwargs['return_set']:
    #     return _return_result(res, return_obj)
    #     return res 