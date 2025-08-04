"""
contains_ - determines if a polytope contains another set or points

This function determines if a polytope contains another set or a collection of points
using exact or approximate methods.

Syntax:
    res, cert, scaling = contains_(P, S, method, tol, maxEval, certToggle, scalingToggle)

Inputs:
    P - polytope object
    S - contSet object or single point or matrix of points
    method - method used for the containment check:
               'exact': Checks for exact containment by looping over halfspaces
               'exact:polymax': Uses halfspace representation
               'exact:venum': Uses vertex representation when applicable
               'approx': Approximative containment check
    tol - tolerance for the containment check; the higher the
       tolerance, the more likely it is that points near the boundary of
       P will be detected as lying in P, which can be useful to
       counteract errors originating from floating point errors.
    maxEval - not relevant for polytope containment
    certToggle - if True, compute certification
    scalingToggle - if True, compute scaling factors

Outputs:
    res - True/False indicating containment
    cert - certification of the result
    scaling - smallest scaling factor needed for containment

Authors: Niklas Kochdumper, Viktor Kotsev, Adrian Kulmburg, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 19-November-2019 (MATLAB)
Last update: 31-October-2024 (TL, added v-polytope/contains) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Tuple, Any, Optional
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from scipy.spatial import ConvexHull # For aux_contains_Vpoly_pointcloud
from cora_python.contSet.polytope.private.priv_equality_to_inequality import priv_equality_to_inequality
from cora_python.contSet.contSet.supportFunc_ import supportFunc_

from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from .polytope import Polytope
    from cora_python.contSet.contSet import ContSet
    from cora_python.contSet.zonotope.zonotope import Zonotope # For aux_exactParser

# Placeholder for priv_compact_alignedEq - this will need to be properly imported or implemented
# Placeholder for priv_normalizeConstraints - this will need to be properly imported or implemented


def contains_(P: 'Polytope', S: Union[np.ndarray, 'ContSet'], method: str = 'exact', 
              tol: float = 1e-12, maxEval: int = 0, certToggle: bool = True, 
              scalingToggle: bool = False) -> Tuple[Union[bool, np.ndarray], bool, Union[float, np.ndarray]]:
    """
    Determine if polytope contains another set or points
    
    Args:
        P: Polytope object
        S: Set or points to check for containment
        method: Containment check method
        tol: Tolerance for containment check
        maxEval: Maximum evaluations (not used for polytopes)
        certToggle: Whether to compute certification
        scalingToggle: Whether to compute scaling factors
        
    Returns:
        Tuple of (result, certification, scaling)
    """

    # Special cases for empty sets
    if P.representsa_('emptySet', 0):
        # An empty set contains nothing unless S is also an empty set.
        if isinstance(S, np.ndarray):
            if S.size == 0: # Empty point cloud
                return np.array([True], dtype=bool), np.array([True], dtype=bool), np.array([0.0])
            else:
                return np.full(S.shape[1], False, dtype=bool), np.full(S.shape[1], True, dtype=bool), np.full(S.shape[1], np.inf)
        elif S.representsa_('emptySet', 0):
            return True, True, 0.0
        else:
            return False, True, np.inf

    # Special cases for fullspace
    if P.representsa_('fullspace', 0):
        # A fullspace contains everything that has the same dimension.
        # If S is fullspace, it is contained. If S is not fullspace, it is still contained if it's a valid set/point in the same dimension.
        if isinstance(S, np.ndarray):
            if S.shape[0] != P.dim():
                 # Incompatible dimensions for point cloud and fullspace
                return np.full(S.shape[1], False, dtype=bool), np.full(S.shape[1], True, dtype=bool), np.full(S.shape[1], np.inf)
            return np.full(S.shape[1], True, dtype=bool), np.full(S.shape[1], True, dtype=bool), np.full(S.shape[1], 0.0)
        elif hasattr(S, 'dim') and S.dim() == P.dim():
            return True, True, 0.0 # Set always contained in fullspace of same dim
        else:
            # If S is not a NumPy array and has different dimensions, it cannot be contained.
            return False, True, np.inf

    # Point cloud in polytope containment
    if isinstance(S, np.ndarray):
        if S.shape[0] != P.dim():
            raise CORAerror('CORA:wrongDimension', f'Dimension of point cloud ({S.shape[0]}) must match polytope dimension ({P.dim()}).')

        # If S is a single point, convert it to a 2D array for consistent processing
        if S.ndim == 1:
            S = S.reshape(-1, 1)
        
        # Use _aux_exactParser to determine the correct underlying function
        # For point clouds, we always use H-poly check unless V-poly is forced
        if method == 'exact:venum':
            # Force V-representation for P, then use V-poly point cloud check
            # This might involve converting P to V-rep if not already
            P.vertices_()
            res, cert, scaling = _aux_contains_Vpoly_pointcloud(P, S, tol, scalingToggle)
        else: # Default to H-poly point cloud check
            # This might involve converting P to H-rep if not already
            P.constraints()
            res, cert, scaling = _aux_contains_Hpoly_pointcloud(P, S, tol, scalingToggle)

        return res, cert, scaling

    # Now we know that S is a ContSet
    # First, deal with trivial cases (already handled emptySet/fullspace above)
    # Try to determine if S represents a single point
    # Safely unpack, or get default None if not a tuple
    representsa_result = S.representsa_('point')
    if isinstance(representsa_result, tuple) and len(representsa_result) == 2:
        is_point, point_val = representsa_result
    else:
        is_point = representsa_result # Assume it's just the boolean result
        point_val = None # No point value available

    if is_point:
        # If S is a single point, check if it's contained in P
        res_pc, cert_pc, scaling_pc = contains_(P, point_val.reshape(-1, 1), 'exact', tol, maxEval, certToggle, scalingToggle)
        # Ensure single boolean/float output for single point sets
        return bool(res_pc), bool(cert_pc), float(scaling_pc) 

    # 1D -> cheap computation of vertices (skip linear program below)
    if P.dim() == 1:
        # Convert S to its vertices and check containment
        try:
            # Try to get vertices directly if available
            if hasattr(S, 'vertices_') and callable(S.vertices_):
                S_vertices = S.vertices_()
            elif hasattr(S, 'vertices') and callable(S.vertices):
                S_vertices = S.vertices()
            else:
                raise AttributeError("S has no 'vertices' or 'vertices_' method for 1D conversion")

            if S_vertices.size == 0: # If the set is empty (after vertex conversion)
                return True, True, 0.0

            # Recursively call contains_ with the point cloud of vertices
            res_pc, cert_pc, scaling_pc = contains_(P, S_vertices, 'exact', tol, maxEval, certToggle, scalingToggle)
            res = np.all(res_pc) # Aggregate to single boolean
            cert = True
            scaling = np.max(scaling_pc) if scaling_pc.size > 0 else 0.0 # Take max scaling
            return res, cert, scaling
        except Exception as e: 
            # Fallback for sets that don't have well-defined vertices or other issues
            # If this happens, proceed to aux_exactParser
            pass

    # Now choose which algorithm to use for contSet objects
    # MATLAB uses a switch on `method` and `class(S)`.
    # Python equivalent will use if/elif chain and isinstance checks.
    
    res, cert, scaling = _aux_exactParser(P, S, method, tol, maxEval, certToggle, scalingToggle)
    # Ensure single boolean/float output for set-set containment (non-point cloud)
    if isinstance(res, np.ndarray):
        res = np.all(res) # Aggregate array to single boolean
    if isinstance(cert, np.ndarray):
        cert = np.all(cert) # Aggregate array to single boolean
    if isinstance(scaling, np.ndarray) and scaling.size > 0:
        scaling = np.max(scaling) # Aggregate array to single float
    elif isinstance(scaling, np.ndarray) and scaling.size == 0:
        scaling = 0.0 # Empty array for scaling implies 0.0 (e.g., empty set)
    
    return bool(res), bool(cert), float(scaling)


# Auxiliary functions -----------------------------------------------------

def _aux_exactParser(P: 'Polytope', S: 'ContSet', method: str, tol: float, maxEval: int, certToggle: bool, scalingToggle: bool) -> Tuple[Union[bool, np.ndarray], bool, Union[float, np.ndarray]]:
    """
    Chooses what algorithm to call upon to check for exact containment.
    
    Args:
        P: Polytope object
        S: ContSet object
        method: Exact method ('exact', 'exact:venum', 'exact:polymax')
        tol: Tolerance
        maxEval: Not used
        certToggle: Compute certification
        scalingToggle: Compute scaling
        
    Returns:
        Tuple of (result, certification, scaling)
    """
    cert = True
    res, scaling = False, np.inf # Initialize

    # Handle specific types of ContSet for 'exact' methods
    # This matches MATLAB's switch based on class(S)
    from cora_python.contSet.polytope.polytope import Polytope # Local import to avoid circular dependency
    from cora_python.contSet.zonotope.zonotope import Zonotope # Placeholder import for Zonotope
    # from cora_python.contSet.conZonotope.conZonotope import ConZonotope # Placeholder import
    # from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle # Placeholder import
    # from cora_python.contSet.capsule.capsule import Capsule # Placeholder import
    # from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid # Placeholder import
    # from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow # Placeholder import
    # from cora_python.contSet.interval.interval import Interval # Placeholder import
    # from cora_python.contSet.halfspace.halfspace import Halfspace # Placeholder import
    # from cora_python.contSet.conHyperplane.conHyperplane import ConHyperplane # Placeholder import


    if isinstance(S, Polytope):
        # For Polytope in Polytope containment
        if method == 'exact':
            if P.isHRep:
                res, scaling = _aux_contains_P_Hpoly(P, S, tol, scalingToggle, certToggle)
            else:
                res, scaling = _aux_contains_P_Vpoly(P, S, tol, scalingToggle, certToggle)
        elif method == 'exact:venum':
            P.vertices_() # Force V-representation
            res, scaling = _aux_contains_P_Vpoly(P, S, tol, scalingToggle, certToggle)
        elif method == 'exact:polymax':
            P.constraints() # Force H-representation
            res, scaling = _aux_contains_P_Hpoly(P, S, tol, scalingToggle, certToggle)
    # elif isinstance(S, Zonotope):
    #     # Handle Zonotope containment, similar logic as Polytope
    #     pass
    # ... add other set types here as they are implemented
    else:
        raise CORAerror('CORA:noExactAlg', P, S) # No exact algorithm for this type of set

    return res, cert, scaling


def _aux_contains_Hpoly_pointcloud(P: 'Polytope', V: np.ndarray, tol: float, scalingToggle: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check containment of a point cloud in an H-polytope.
    This function is based on MATLAB's aux_contains_Hpoly_pointcloud.
    
    Args:
        P: Polytope object (H-representation).
        V: Point cloud (vertices) as a NumPy array (n x num_points).
        tol: Tolerance.
        scalingToggle: Whether to compute scaling factors.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (results_bool_array, certification_array, scaling_array)
    """
    # Ensure P is in H-representation, if not, convert it
    if not P.isHRep:
        P.constraints() # Convert to H-representation

    A = P.A
    b = P.b
    Ae = P.Ae
    be = P.be

    num_points = V.shape[1]
    res = np.full(num_points, True, dtype=bool)
    scaling = np.zeros(num_points)
    cert = np.full(num_points, True, dtype=bool) # Certification is not fully implemented
    
    # Check inequality constraints: A*V <= b
    if A.size > 0 and V.size > 0:
        violations_ineq = A @ V - b # A (m x n), V (n x num_points), b (m x 1)
        # A*V should be <= b, so violations should be <= 0
        # Use withinTol for nearly zero values
        # MATLAB uses (violations > tol) & (~withinTol(violations, 0, tol))
        # This means, if violation is greater than tol and not within tolerance of zero
        # it's a violation. We want: violations > tol. Or, violations <= 0 + tol.
        # So, if violations_ineq[i,j] > tol for some j, then it's a violation.
        # Or, we want (violations_ineq < tol) for all elements.
        # More simply: check if max(A*V - b) <= tol

        # Find points that violate any inequality constraint
        # A point is contained if all its entries satisfy the constraints
        # Each row of violations_ineq corresponds to a constraint. Each column to a point.
        for i in range(violations_ineq.shape[0]):
            # Check if any point violates this constraint
            current_violations = violations_ineq[i, :]
            violating_indices = current_violations > tol
            
            if np.any(violating_indices):
                res[violating_indices] = False
                # Compute scaling for violated points
                current_scaling = current_violations[violating_indices] / (b[i,0] + tol) # Avoid div by zero
                scaling[violating_indices] = np.maximum(scaling[violating_indices], current_scaling)

    # Check equality constraints: Ae*V = be
    if Ae.size > 0 and V.size > 0:
        violations_eq = Ae @ V - be
        # For equalities, check if abs(Ae*V - be) > tol
        # Find points that violate any equality constraint
        for i in range(violations_eq.shape[0]):
            current_violations = np.abs(violations_eq[i, :])
            violating_indices = current_violations > tol
            
            if np.any(violating_indices):
                res[violating_indices] = False
                # For equality violations, scaling is usually not defined or is Inf
                scaling[violating_indices] = np.inf

    return res, cert, scaling


def _aux_contains_Vpoly_pointcloud(P: 'Polytope', S: np.ndarray, tol: float, scalingToggle: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check containment of point cloud in V-polytope using ConvexHull (equivalent to Quickhull).
    
    Args:
        P: Polytope object (assumed to be in V-representation)
        S: Points to check (dim x num_points matrix)
        tol: Tolerance
        scalingToggle: Not supported by this method, will return default values.
        
    Returns:
        Tuple of (result, certification, scaling)
    """

    # Scaling is not directly computed by this method, set defaults
    num_points = S.shape[1]
    scaling = np.full(num_points, np.inf)
    cert = np.ones(num_points, dtype=bool) # Results are always exact with convex hull

    n = P.dim()

    # Special method for 1D
    if n == 1:
        V_min = np.min(P.V)
        V_max = np.max(P.V)
        # Check if points are within the interval [V_min, V_max]
        res = (S >= V_min - tol) & (S <= V_max + tol) # Apply tolerance
        return res.flatten(), cert.flatten(), scaling # Ensure 1D array for res and cert
    
    # Ensure P.V and S are in (num_points, dim) format for ConvexHull
    # ConvexHull expects (num_points, dim)
    P_V_hull = P.V.T # Transpose (dim x num_vertices) to (num_vertices x dim)
    S_hull = S.T # Transpose (dim x num_points) to (num_points x dim)

    res = np.zeros(num_points, dtype=bool)

    # If the polytope is empty, no points are contained
    if P.isemptyobject():
        return res, cert, scaling

    # Create a ConvexHull object from the polytope's vertices
    try:
        # Handle degenerate cases for ConvexHull (e.g., single point, collinear points)
        if P_V_hull.shape[0] < n + 1: # Not enough points for a full-dimensional hull
            if P_V_hull.shape[0] == 1: # Single point polytope
                for i in range(num_points):
                    # Check if the point S[:,i] is approximately equal to the single vertex P_V_hull[0]
                    res[i] = np.allclose(S_hull[i], P_V_hull[0], atol=tol)
            else: # Collinear points in higher dimension or other degenerate case
                # Fallback to H-representation if V-rep is degenerate and not a single point
                P.constraints() # Convert to H-representation
                return _aux_contains_Hpoly_pointcloud(P, S, tol, scalingToggle)
        
        hull = ConvexHull(P_V_hull)
        
        # Check each point for containment
        for i in range(num_points):
            point = S_hull[i]
            # Use `point_in_simplex` or similar geometric check.
            # `hull.equations` provides A and b for A*x <= b.
            # Ax - b <= 0. If Ax - b > tol, it's outside.
            # Equations are (normals, offsets)
            violations = (hull.equations[:, :-1] @ point) + hull.equations[:, -1]
            res[i] = np.all(violations <= tol) # True if all violations are within tolerance
            
    except Exception as e:
        # If convhulln fails (e.g., due to degenerate points), fall back to checking if points are vertices
        # This part is simplified from MATLAB's linprog fallback in aux_contains_Vpoly_pointcloud
        # For simplicity, we'll only check if the points are vertices for now.
        # A more robust solution would involve a custom LP solver for containment.
        from scipy.optimize import linprog
        for i in range(num_points):
            # Check if the point S[:,i] is in the convex hull of P.V using linear programming
            # min 1 (arbitrary objective function)
            # s.t. P.V * beta = S[:,i]
            #      sum(beta) = 1
            #      beta >= 0
            
            # For linprog: min c.T * x
            # s.t. A_ub @ x <= b_ub
            #      A_eq @ x == b_eq
            #      bounds

            # Number of vertices (variables for beta)
            num_vertices = P_V_hull.shape[0]
            
            # Objective function: min 1 (any constant will do)
            c = np.ones(num_vertices)
            
            # Equality constraints: P.V * beta = S[:,i] and sum(beta) = 1
            # A_eq matrix for [V.T; ones(1, num_vertices)]
            # For linprog, V is (num_vertices x dim), point is (dim)
            A_eq = np.vstack([P_V_hull.T, np.ones((1, num_vertices))]) # Transpose P_V_hull to match V * beta
            b_eq = np.vstack([S_hull[i].reshape(-1,1), np.array([[1.0]])])

            # Bounds: beta >= 0
            bounds = (0, None) # (min, max) for each variable

            # Solve the linear program
            # Using 'highs-ds' solver for better performance and robustness with degenerate problems
            res_lp = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds', options={'tol': tol})

            # If the LP is feasible, the point is contained
            res[i] = res_lp.success

    return res, cert, scaling

def _aux_contains_P_Hpoly(P: 'Polytope', S: 'ContSet', tol: float, scalingToggle: bool, certToggle: bool) -> Tuple[bool, float]:
    """
    Containment check for any set in H-polytope using support functions.
    Assumes P is in H-representation.
    
    Args:
        P: Polytope object (H-representation)
        S: ContSet object
        tol: Tolerance
        scalingToggle: Compute scaling
        certToggle: Compute certification
        
    Returns:
        Tuple of (result, scaling)
    """
    from cora_python.contSet.polytope.polytope import Polytope # Explicitly import Polytope here

    # If scaling has to be computed, first need to shift center to origin
    if scalingToggle:
        c = P.center()
        P_shifted = P - c
        S_shifted = S - c
    else:
        P_shifted = P
        S_shifted = S

    # Check if S is a Polytope in V-representation: fast method
    if isinstance(S_shifted, Polytope) and S_shifted.isVRep:
        # Get vertices of S and call point cloud containment for H-polytope
        res_pc, _, scaling_pc = _aux_contains_Hpoly_pointcloud(P_shifted, S_shifted.V, tol, scalingToggle)
        res = np.all(res_pc)
        scaling = np.max(scaling_pc) if scaling_pc.size > 0 else 0.0
        return res, scaling

    # Generic method: check support function value along each normal vector of
    # equality and inequality constraints
    # Get combined A, b from P.A, P.b, P.Ae, P.be
    combined_A, combined_b = priv_equality_to_inequality(P_shifted.A, P_shifted.b, P_shifted.Ae, P_shifted.be)
    
    scaling = 0.0
    res = True

    # Loop over all constraints
    for i in range(combined_b.shape[0]):
        normal_vector = combined_A[i,:].reshape(-1,1)
        b_polytope = combined_b[i,0]
        
        # Compute support function of S along the normal vector
        # The MATLAB code uses 'upper' for supportFunc_. We assume the same here.
        b_set_sup_val, _ = supportFunc_(S_shifted, normal_vector, 'upper') # Unpack the tuple
    
        # Check for containment condition: sup(S, l) <= sup(P, l)
        # Account for tolerance: b_set_sup > b_polytope + tol
        if b_set_sup_val > b_polytope and not withinTol(b_polytope, b_set_sup_val, tol):
            res = False
            scaling_current = b_set_sup_val / b_polytope # Ratio of support functions
            if scaling_current > scaling:
                scaling = scaling_current
            # Early exit for no certification
            if not certToggle:
                return res, scaling
    return res, scaling


def _aux_contains_P_Vpoly(P: 'Polytope', S: 'ContSet', tol: float, scalingToggle: bool, certToggle: bool) -> Tuple[bool, float]:
    """
    Containment check for any set in V-polytope using support functions (dual method).
    Assumes P is in V-representation.
    
    Args:
        P: Polytope object (V-representation)
        S: ContSet object
        tol: Tolerance
        scalingToggle: Compute scaling
        certToggle: Compute certification
        
    Returns:
        Tuple of (result, scaling)
    """
    from cora_python.contSet.polytope.polytope import Polytope # Explicitly import Polytope here

    # If scaling has to be computed, first need to shift center to origin
    if scalingToggle:
        c = P.center()
        P_shifted = P - c
        S_shifted = S - c
    else:
        P_shifted = P
        S_shifted = S

    # Convert P_shifted to H-representation for the support function method
    # This is a key step as the V-polytope containment check often involves
    # converting to H-rep and then checking support functions.
    P_shifted.constraints() # Convert to H-representation

    # Now, call the H-polytope containment check
    # This implicitly relies on the _aux_contains_P_Hpoly for the actual logic
    # It's important that _aux_contains_P_Hpoly correctly handles the parameters.
    # Since this function returns (result, scaling), we return them directly.
    # The certToggle is passed to ensure consistency.
    res, scaling = _aux_contains_P_Hpoly(P_shifted, S_shifted, tol, scalingToggle, certToggle)

    return res, scaling 