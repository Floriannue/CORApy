from typing import Union, TYPE_CHECKING
import numpy as np
from cora_python.g.functions.matlab.validate.preprocessing import find_class_arg
from cora_python.g.functions.matlab.validate.check import equal_dim_check
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.contSet.reorder import reorder

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope
    from cora_python.contSet.zonotope.zonotope import Zonotope
    from cora_python.contSet.interval.interval import Interval
    from cora_python.contSet.conZonotope.conZonotope import ConZonotope
    from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle

def _aux_plus_Hpoly_Hpoly(P1: 'Polytope', P2: 'Polytope', n: int) -> 'Polytope':
    from cora_python.contSet.polytope.polytope import Polytope
    from cora_python.contSet.polytope.project import project
    
    A1, b1, Ae1, be1 = P1.A, P1.b, P1.Ae, P1.be
    A2, b2, Ae2, be2 = P2.A, P2.b, P2.Ae, P2.be

    # Handle empty A1/A2 or Ae1/Ae2 to ensure correct shapes for np.block
    # For A, it's (rows, n)
    # For b, it's (rows, 1)

    # Construct the main A block for inequalities
    if A1.size > 0 and A2.size > 0:
        Z_A = np.zeros((A1.shape[0], A2.shape[1])) # Shape of A1 rows x A2 cols
        A = np.block([[A2, -A2], [Z_A, A1]])
        b = np.vstack([b2, b1])
    elif A1.size > 0: # Only P1 has inequality constraints
        A = np.block([np.zeros((A1.shape[0], n)), A1]) # Zero matrix for A2 part
        b = b1
    elif A2.size > 0: # Only P2 has inequality constraints
        A = np.block([A2, np.zeros((A2.shape[0], n))]) # Zero matrix for A1 part
        b = b2
    else: # Both A1 and A2 are empty
        A = np.zeros((0, 2 * n)) # Resulting A is empty, with 2n columns
        b = np.zeros((0, 1))

    Ae = np.zeros((0, 2 * n)) # Default empty Ae for combined equalities
    be = np.zeros((0, 1))    # Default empty be for combined equalities

    # Construct the Ae block for equalities
    if Ae1.size > 0 and Ae2.size > 0:
        Z_Ae = np.zeros((Ae1.shape[0], Ae2.shape[1])) # Shape of Ae1 rows x Ae2 cols
        Ae = np.block([[Ae2, -Ae2], [Z_Ae, Ae1]])
        be = np.vstack([be2, be1])
    elif Ae1.size > 0: # Only P1 has equality constraints
        Ae = np.block([np.zeros((Ae1.shape[0], n)), Ae1])
        be = be1
    elif Ae2.size > 0: # Only P2 has equality constraints
        Ae = np.block([Ae2, np.zeros((Ae2.shape[0], n))])
        be = be2
    # If both Ae1 and Ae2 are empty, Ae and be remain the default empty arrays

    P_highdim = Polytope(A, b, Ae, be)

    return project(P_highdim, list(range(1, n + 1)))

def _aux_plus_Vpoly_Vpoly(P1: 'Polytope', P2: 'Polytope', n: int) -> 'Polytope':
    from cora_python.contSet.polytope.polytope import Polytope
    V1, V2 = P1.V, P2.V

    # Vectorized computation of Minkowski sum of vertices (equivalent to MATLAB's bsxfun)
    # V1 is (n x num_v1), V2 is (n x num_v2)
    # We want V = (n x (num_v1 * num_v2))
    # V[:, j + i*num_v1] = V1[:, j] + V2[:, i]
    # np.newaxis expands dimensions to allow broadcasting
    V = V1[:, :, np.newaxis] + V2[:, np.newaxis, :]
    V = V.reshape(n, -1) # Reshape to (n x total_num_vertices)
        
    return Polytope(V)

def _aux_plus_poly_point(P: 'Polytope', S: np.ndarray) -> 'Polytope':
    from cora_python.contSet.polytope.private.priv_plus_minus_vector import priv_plus_minus_vector
    from cora_python.contSet.polytope.polytope import Polytope
    if S.shape[1] > 1:
        raise CORAerror('CORA:noops', P, S)
    if S.shape[0] != P.dim():
        raise CORAerror('CORA:notSupported',
                        'Minkowski addition with scalar is not supported unless the set is 1-dimensional.')
    
    # Use H-representation if available, otherwise convert
    if P.isHRep:
        A, b, Ae, be = priv_plus_minus_vector(P.A, P.b, P.Ae, P.be, S)
        P_out = Polytope(A, b, Ae, be)
    elif P.isVRep:
        # For vertex representation, just add the vector to all vertices
        V_new = P.V + S # Broadcasting should handle this
        P_out = Polytope(V_new)
    else:
        # This case should not be reached with the new constructor
        # Force computation of H-rep and use that
        from .constraints import constraints
        P_H = constraints(P)
        A, b, Ae, be = priv_plus_minus_vector(P_H.A, P_H.b, P_H.Ae, P_H.be, S)
        P_out = Polytope(A, b, Ae, be)
        
    return P_out

def _aux_setproperties(P_out: 'Polytope', P_in1: 'Polytope', P_in2: 'Polytope') -> 'Polytope':
    """
    Infers and sets set properties (bounded, fullDim) for the output polytope.
    Matches MATLAB's aux_setproperties.m.
    """
    # If both input polytopes are bounded, then the sum is also bounded
    if P_in1.bounded and P_in2.bounded:
        P_out._bounded_val = True
        P_out._bounded_is_computed = True
    # If one of the input polytopes is unbounded, then the sum is also unbounded
    elif not P_in1.bounded or not P_in2.bounded: # Assuming 'bounded' property returns actual boolean or triggers computation
        P_out._bounded_val = False
        P_out._bounded_is_computed = True
    
    # If one of the input polytopes is full-dimensional, the sum is, too
    if P_in1.fullDim or P_in2.fullDim:
        P_out._fullDim_val = True
        P_out._fullDim_is_computed = True

    return P_out

def plus(p1: Union['Polytope', np.ndarray], p2: Union['Polytope', np.ndarray]) -> 'Polytope':
    
    p1, p2 = reorder(p1, p2)
    
    from cora_python.contSet.contSet.contSet import ContSet
    if isinstance(p2, ContSet) and p2.precedence < p1.precedence:
        return p2 + p1
        
    equal_dim_check(p1, p2)
    n = p1.dim()
    tol = 1e-10
    
    from cora_python.contSet.polytope.polytope import Polytope
    
    if isinstance(p2, Polytope):
        # Check representation flags BEFORE representsa calls that might change them
        has_v1_orig = p1.isVRep
        has_v2_orig = p2.isVRep
        has_h1_orig = p1.isHRep
        has_h2_orig = p2.isHRep
        
        # Special case checks
        if p1.representsa('fullspace', tol) or p2.representsa('fullspace', tol):
            return Polytope.Inf(n)
        if p1.representsa('emptySet', tol) or p2.representsa('emptySet', tol):
            return Polytope.empty(n)
        if p1.representsa('origin', tol):
            return p2
        if p2.representsa('origin', tol):
            return p1
        
        # Use the original representation flags for path determination
        # Prioritize V-representation over H-representation for efficiency
        if has_v1_orig and has_v2_orig:
            s_out = _aux_plus_Vpoly_Vpoly(p1, p2, n)
        elif has_h1_orig and has_h2_orig:
            s_out = _aux_plus_Hpoly_Hpoly(p1, p2, n)
        else:
            # Force conversion to H-representation
            from .constraints import constraints
            p1_H = constraints(p1)
            p2_H = constraints(p2)
            s_out = _aux_plus_Hpoly_Hpoly(p1_H, p2_H, n)
            
        s_out = _aux_setproperties(s_out, p1, p2) # Call new helper to infer properties
        return s_out
    
    # Check fullspace for non-polytope p2
    if p1.representsa('fullspace', tol) or (hasattr(p2, 'representsa') and p2.representsa('fullspace', tol)):
        return Polytope.Inf(n)
    
    if isinstance(p2, np.ndarray) and p2.ndim == 2 and p2.shape[1] == 1:
        s_out = _aux_plus_poly_point(p1, p2)
        return s_out
        
    # Other set types
    if type(p2).__name__ in ['Zonotope', 'Interval', 'ConZonotope', 'ZonoBundle']:
        s_poly = Polytope(p2)
        return p1 + s_poly
        
    raise CORAerror('CORA:noops', p1, p2) 