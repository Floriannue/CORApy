from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from cora_python.contSet.polytope.polytope import Polytope # Add this import
from typing import Union

if TYPE_CHECKING:
    from .polytope import Polytope

def minus(p: Polytope, q: Union[np.ndarray, list]) -> Polytope:
    """
    Overloads the '-' operator for polytopes. Only subtraction of a vector
    is supported, which corresponds to a translation of the polytope.
    
    p - q  => p.__sub__(q)

    Args:
        p (Polytope): The polytope.
        q (object): The object to subtract (must be a numpy vector).

    Returns:
        Polytope: The translated polytope.
    """
    # Ensure q is a numpy array and has compatible dimensions
    if not (isinstance(q, np.ndarray) and q.ndim <= 2 and 
            (q.shape[0] == p.dim() or (q.ndim == 1 and q.size == p.dim()))): # Added check for 1D array q
        # Minkowski difference is not supported via '-'
        raise TypeError(
            "The '-' operator for a Polytope is only defined for subtraction "
            "of a numerical vector with compatible dimensions. For Minkowski difference, use minkDiff()."
        )
    
    # Ensure q is a column vector for consistent matrix operations
    q_vec = q.reshape(-1, 1)

    # Create a deep copy to not modify the original
    p_new = Polytope(p)

    # Translate H-representation if it exists
    if p_new.isHRep:
        # Correct subtraction for H-representation: b_new = b - A @ q_vec
        # For P - q: if Ax <= b, then A(x' + q) <= b where x' = x - q
        # This gives Ax' <= b - Aq, so b_new = b - Aq
        if p_new.b is not None and p_new.A is not None and p_new.A.size > 0:
            p_new._b = p_new.b - p_new.A @ q_vec  # b_new = b - A*q for P - q
        # Only adjust be if equality constraints exist (keeps None semantics out; arrays guaranteed by constructor)
        if p_new.Ae is not None and p_new.Ae.size > 0 and p_new.be is not None and p_new.be.size > 0:
            p_new._be = p_new.be - p_new.Ae @ q_vec
        p_new._reset_lazy_flags() # Reset lazy flags after modifying H-representation
    
    # Translate V-representation if it exists
    if p_new.isVRep:
        # Correct subtraction for V-representation: V_new = V - q_vec
        p_new._V = p_new.V - q_vec # Broadcasting should handle this correctly
        p_new._reset_lazy_flags() # Reset lazy flags after modifying V-representation

    return p_new

def rminus(p: Polytope, q: object) -> Polytope:
    """
    Overloads the '-' operator for right-sided subtraction. Not supported.
    
    q - p => p.__rsub__(q)
    """
    raise TypeError("Right-sided subtraction is not supported for Polytopes.") 