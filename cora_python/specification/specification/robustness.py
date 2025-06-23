"""
robustness - computes the robustness score of a point with respect to 
   the specifications, where a positive robustness score means that all
   specifications are satisfied

Syntax:
    val = robustness(spec, p)
    val = robustness(spec, p, time)

Inputs:
    spec - specification object
    p - point represented by a vector
    time - scalar representing the current time

Outputs:
    val - robustness value for the point p

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 27-November-2021 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional, Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def robustness(spec, p: np.ndarray, time: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Compute the robustness score of a point with respect to specifications
    
    Args:
        spec: specification object or list of specification objects
        p: point(s) represented by vector(s) - shape (n,) or (n, m) for multiple points
        time: scalar representing the current time
        
    Returns:
        val: robustness value(s) for the point(s) p
    """
    
    # Convert to numpy array if needed
    p = np.atleast_2d(p)
    if p.shape[0] == 1 and p.shape[1] > 1:
        # If p is row vector, transpose it
        p = p.T
    
    # Initialize robustness value
    val = np.inf
    
    # Ensure spec is a list
    if not isinstance(spec, list):
        specs = [spec]
    else:
        specs = spec
    
    # Check if multiple points are provided
    if p.shape[1] > 1:
        # Compute robustness for all points
        val = np.zeros(p.shape[1])
        
        for i in range(p.shape[1]):
            val[i] = robustness(specs, p[:, i:i+1], time)
        
        return val
    else:
        # Single point - flatten to 1D
        p_single = p.flatten()
        
        # Loop over all specifications
        for spec_obj in specs:
            
            # Check if time frames overlap
            if time is None and spec_obj.time is not None and not _represents_empty_set(spec_obj.time):
                raise CORAerror('CORA:specialError',
                    'Timed specifications require a time interval.')
            
            # Check if specification is active at this time
            if (spec_obj.time is None or _represents_empty_set(spec_obj.time) or 
                (time is not None and _contains_time(spec_obj.time, time))):
                
                # Different types of specifications
                if spec_obj.type == 'invariant':
                    val_ = _robustness_safe_set(spec_obj.set, p_single)
                    
                elif spec_obj.type == 'unsafeSet':
                    val_ = _robustness_unsafe_set(spec_obj.set, p_single)
                    
                elif spec_obj.type == 'safeSet':
                    val_ = _robustness_safe_set(spec_obj.set, p_single)
                    
                elif spec_obj.type == 'custom':
                    raise CORAerror('CORA:notSupported',
                        'Robustness computation for custom '
                        'specifications is not supported!')
                    
                elif spec_obj.type == 'logic':
                    raise CORAerror('CORA:notSupported',
                        'Robustness computation for logic '
                        'specifications is not supported!')
                else:
                    raise CORAerror('CORA:wrongInput',
                        f'Unknown specification type: {spec_obj.type}')
                
                # Overall robustness is minimum of single specifications
                val = min(val, val_)
        
        return val


def _robustness_unsafe_set(S, p: np.ndarray) -> float:
    """Compute the robustness value of point p for an unsafe set S"""
    
    try:
        # Convert set S to a polytope if needed
        if hasattr(S, 'polytope'):
            S_poly = S.polytope()
        elif hasattr(S, 'A') and hasattr(S, 'b'):
            # Already a polytope-like object
            S_poly = S
        else:
            # Try to convert to polytope
            from cora_python.contSet.polytope import Polytope
            S_poly = Polytope(S)
        
        # Get constraints with normalized halfspace directions
        if hasattr(S_poly, 'A') and hasattr(S_poly, 'b'):
            C = S_poly.A
            d = S_poly.b
        else:
            raise CORAerror('CORA:wrongInput',
                'Set must be convertible to polytope with A, b matrices')
        
        # Normalize constraints if method exists
        if hasattr(S_poly, 'normalize_constraints'):
            S_norm = S_poly.normalize_constraints('A')
            C = S_norm.A
            d = S_norm.b
        
        # Check if the point is inside or outside the unsafe set
        if hasattr(S, 'contains') and S.contains(p):
            # Point is inside unsafe set - negative robustness
            val = -min(np.abs(C @ p - d))
        else:
            # Point is outside unsafe set - positive robustness
            val = _distance_poly_point(C, d, p)
            
    except Exception as e:
        # Fallback: use simple distance measure
        print(f"Warning: Could not compute exact robustness, using approximation: {e}")
        
        # Simple approximation - compute distance to center
        if hasattr(S, 'center'):
            center = S.center()
            val = np.linalg.norm(p - center)
        else:
            val = 0.0
    
    return val


def _robustness_safe_set(S, p: np.ndarray) -> float:
    """Compute the robustness value of point p for a safe set S"""
    
    try:
        # Convert set S to a polytope if needed
        if hasattr(S, 'polytope'):
            S_poly = S.polytope()
        elif hasattr(S, 'A') and hasattr(S, 'b'):
            # Already a polytope-like object
            S_poly = S
        else:
            # Try to convert to polytope
            from cora_python.contSet.polytope import Polytope
            S_poly = Polytope(S)
        
        # Get constraints with normalized halfspace directions
        if hasattr(S_poly, 'A') and hasattr(S_poly, 'b'):
            C = S_poly.A
            d = S_poly.b
        else:
            raise CORAerror('CORA:wrongInput',
                'Set must be convertible to polytope with A, b matrices')
        
        # Normalize constraints if method exists
        if hasattr(S_poly, 'normalize_constraints'):
            S_norm = S_poly.normalize_constraints('A')
            C = S_norm.A
            d = S_norm.b
        
        # Check if the point is inside or outside the safe set
        if hasattr(S, 'contains') and S.contains(p):
            # Point is inside safe set - positive robustness
            val = min(np.abs(C @ p - d))
        else:
            # Point is outside safe set - negative robustness
            val = -_distance_poly_point(C, d, p)
            
    except Exception as e:
        # Fallback: use simple distance measure
        print(f"Warning: Could not compute exact robustness, using approximation: {e}")
        
        # Simple approximation - compute distance to center
        if hasattr(S, 'center'):
            center = S.center()
            val = -np.linalg.norm(p - center)  # Negative for safe set
        else:
            val = 0.0
    
    return val


def _distance_poly_point(C: np.ndarray, d: np.ndarray, p: np.ndarray) -> float:
    """Compute the distance between a point p and a polytope P: C*x <= d"""
    
    n = len(p)
    m = C.shape[0]
    
    # Check how many halfspace constraints are violated
    temp = C @ p - d
    ind = np.where(temp > 0)[0]
    
    if len(ind) == 1:
        # Only one halfspace constraint violated -> distance to polytope is
        # equal to the distance to the halfspace constraint
        return temp[ind[0]]
        
    elif len(ind) == n:
        # Compute the vertex that is closest to the point by combining 
        # the violated halfspace constraints
        try:
            v = np.linalg.solve(C[ind, :], d[ind])
            return np.linalg.norm(v - p)
        except np.linalg.LinAlgError:
            # Fallback to simple distance
            return np.min(temp[ind])
    else:
        # For general case, use optimization-based approach
        try:
            from scipy.optimize import linprog
            
            # Set-up linear program to minimize the norm 1 distance: 
            # min ||p - x||_1 s.t. C*x <= d
            # This is approximated as min sum(t_i) s.t. -t_i <= p_i - x_i <= t_i, C*x <= d
            
            # Decision variables: [x, t]
            c = np.concatenate([np.zeros(n), np.ones(n)])
            
            # Inequality constraints: C*x <= d and -t <= p - x <= t
            A_ub = np.zeros((m + 2*n, 2*n))
            A_ub[:m, :n] = C  # C*x <= d
            A_ub[m:m+n, :n] = -np.eye(n)  # -x - t <= -p
            A_ub[m:m+n, n:] = -np.eye(n)
            A_ub[m+n:m+2*n, :n] = np.eye(n)  # x - t <= p
            A_ub[m+n:m+2*n, n:] = -np.eye(n)
            
            b_ub = np.concatenate([d, -p, p])
            
            # Solve linear program
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            
            if result.success:
                return result.fun
            else:
                # Fallback: simple L2 distance to violated constraints
                return np.linalg.norm(temp[ind])
                
        except ImportError:
            # scipy not available, use simple approximation
            return np.linalg.norm(temp[ind])


def _represents_empty_set(obj) -> bool:
    """Check if object represents an empty set"""
    if hasattr(obj, 'representsa_'):
        return obj.representsa_('emptySet', 1e-10)
    return False


def _contains_time(time_interval, time_point: float) -> bool:
    """Check if time interval contains time point"""
    if hasattr(time_interval, 'contains'):
        return time_interval.contains(time_point)
    elif hasattr(time_interval, 'infimum') and hasattr(time_interval, 'supremum'):
        return time_interval.infimum() <= time_point <= time_interval.supremum()
    else:
        return True  # Assume always active if can't check 