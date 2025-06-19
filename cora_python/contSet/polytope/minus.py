from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .polytope import Polytope

def minus(p: Polytope, q: object) -> Polytope:
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
    if isinstance(q, np.ndarray) and q.ndim <= 2 and (q.shape[0] == p.dim() or q.shape[1] == p.dim()):
        
        # Ensure q is a column vector
        q_vec = q.reshape(-1, 1)

        # Create a deep copy to not modify the original
        p_new = p.copy()

        # Translate H-representation if it exists
        if p_new._has_h_rep:
            if p_new._A is not None:
                p_new._b = p_new._b + p_new._A @ q_vec
            if p_new._Ae is not None:
                p_new._be = p_new._be + p_new._Ae @ q_vec
        
        # Translate V-representation if it exists
        if p_new._has_v_rep:
            if p_new._V is not None and p_new._V.size > 0:
                p_new._V = p_new._V - q_vec

        return p_new

    else:
        # Minkowski difference is not supported via '-'
        raise TypeError(
            "The '-' operator for a Polytope is only defined for subtraction "
            "of a numerical vector. For Minkowski difference, use minkDiff()."
        )

def rminus(p: Polytope, q: object) -> Polytope:
    """
    Overloads the '-' operator for right-sided subtraction. Not supported.
    
    q - p => p.__rsub__(q)
    """
    raise TypeError("Right-sided subtraction is not supported for Polytopes.") 