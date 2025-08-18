import numpy as np
from typing import Optional, Tuple
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from .priv_compact_1D import priv_compact_1D
from .priv_normalize_constraints import priv_normalize_constraints

def priv_vertices_1D(A: Optional[np.ndarray], b: Optional[np.ndarray], 
                     Ae: Optional[np.ndarray], be: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
    """
    Compute vertices for 1D polytope following MATLAB logic exactly.
    
    Args:
        A: inequality constraint matrix (n_ineq x 1)
        b: inequality constraint offset (n_ineq,)
        Ae: equality constraint matrix (n_eq x 1)
        be: equality constraint offset (n_eq,)
        
    Returns:
        V: vertices as (1 x n_vertices) array
        empty: true/false whether polytope is empty
    """
    # Debug output
    print(f"DEBUG priv_vertices_1D: A={A}, A.size={A.size if A is not None else 'None'}")
    print(f"DEBUG priv_vertices_1D: b={b}, b.size={b.size if b is not None else 'None'}")
    print(f"DEBUG priv_vertices_1D: Ae={Ae}, Ae.size={Ae.size if Ae is not None else 'None'}")
    print(f"DEBUG priv_vertices_1D: be={be}, be.size={be.size if be is not None else 'None'}")
    
    # Set tolerance like MATLAB
    tol = 1e-12
    
    # Handle empty inputs (MATLAB: zeros(0,1) and zeros(0,0))
    A_norm = A if A is not None and A.size > 0 else np.array([]).reshape(0, 1)
    b_norm = b if b is not None and b.size > 0 else np.array([]).reshape(0, 1)
    Ae_norm = Ae if Ae is not None and Ae.size > 0 else np.array([]).reshape(0, 1)
    be_norm = be if be is not None and be.size > 0 else np.array([]).reshape(0, 1)
    
    # MATLAB: compute minimal representation
    A_compact, b_compact, Ae_compact, be_compact, empty = priv_compact_1D(A_norm, b_norm, Ae_norm, be_norm, tol)
    print(f"DEBUG priv_vertices_1D: after priv_compact_1D: empty={empty}")
    
    # MATLAB: if there is no point, P is already empty
    if empty:
        print(f"DEBUG priv_vertices_1D: polytope is empty")
        return np.array([]).reshape(1, 0), True
    
    # MATLAB: normalize constraints
    A_norm, b_norm, Ae_norm, be_norm = priv_normalize_constraints(A_compact, b_compact, Ae_compact, be_compact, 'A')
    print(f"DEBUG priv_vertices_1D: after priv_normalize_constraints: A={A_norm}, b={b_norm}, Ae={Ae_norm}, be={be_norm}")
    
    # MATLAB: if ~isempty(A)
    if A_norm.size > 0:
        print(f"DEBUG priv_vertices_1D: processing inequality constraints")
        
        # MATLAB: check boundedness from below
        Aisminus1 = withinTol(A_norm, -1, tol)
        if np.any(Aisminus1):
            # bounded from below
            lb_idx = np.where(Aisminus1)[0][0]
            V = np.array([[-b_norm[lb_idx]]])
            print(f"DEBUG priv_vertices_1D: bounded below at {V[0, 0]}")
        else:
            # unbounded toward -Inf
            V = np.array([[-np.inf]])
            print(f"DEBUG priv_vertices_1D: unbounded below")
        
        # MATLAB: check boundedness from above
        Ais1 = withinTol(A_norm, 1, tol)
        if np.any(Ais1):
            # bounded from above (add only if not a duplicate)
            ub_idx = np.where(Ais1)[0][0]
            ub_val = b_norm[ub_idx]
            print(f"DEBUG priv_vertices_1D: bounded above at {ub_val}")
            
            # MATLAB: if ~withinTol(V,b(Ais1))
            if not withinTol(V[0, 0], ub_val, tol):
                V = np.array([[V[0, 0], ub_val]])
            else:
                print(f"DEBUG priv_vertices_1D: upper bound is duplicate, keeping V={V}")
        else:
            # unbounded toward +Inf
            ub_val = np.inf
            print(f"DEBUG priv_vertices_1D: unbounded above")
            V = np.array([[V[0, 0], ub_val]])
        
        print(f"DEBUG priv_vertices_1D: final V from inequalities: {V}")
        return V, False
    
    # MATLAB: elseif ~isempty(Ae)
    elif Ae_norm.size > 0:
        print(f"DEBUG priv_vertices_1D: processing equality constraints")
        # MATLAB: due to minHRep call above, we should only have one equality here
        if Ae_norm.shape[0] == 1:
            V = np.array([[be_norm[0] / Ae_norm[0, 0]]])
            print(f"DEBUG priv_vertices_1D: equality constraint -> single point V={V}")
            return V, False
        else:
            # Multiple equality constraints - this should not happen after priv_compact_1D
            print(f"DEBUG priv_vertices_1D: multiple equality constraints after compact - error")
            raise ValueError("Error in vertex computation of 1D polytope.")
    
    # MATLAB: else (this should not happen if we handled all cases above)
    else:
        print(f"DEBUG priv_vertices_1D: no constraints -> fullspace")
        # This case should be handled by priv_compact_1D, but if we get here, return fullspace
        return np.array([[-np.inf, np.inf]]), False 