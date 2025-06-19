"""
priv_compact_all - removes all redundant constraints in the halfspace representation

Description:
    Removes all redundant constraints from a polytope's halfspace representation
    of an nD polytope by applying various compaction strategies.

Syntax:
    A, b, Ae, be, empty, minHRep = priv_compact_all(A, b, Ae, be, n, tol)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    n - dimension of the polytope
    tol - tolerance

Outputs:
    A - inequality constraint matrix (compacted)
    b - inequality constraint offset (compacted)
    Ae - equality constraint matrix (compacted)
    be - equality constraint offset (compacted)
    empty - true/false whether polytope is empty
    minHRep - minimal representation obtained

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from .priv_compact_zeros import priv_compact_zeros
from .priv_compact_toEquality import priv_compact_toEquality
from .priv_compact_alignedEq import priv_compact_alignedEq
from .priv_compact_alignedIneq import priv_compact_alignedIneq
from .priv_compact_1D import priv_compact_1D
from .priv_compact_2D import priv_compact_2D
from .priv_compact_nD import priv_compact_nD


def priv_compact_all(A, b, Ae, be, n, tol):
    """
    Removes all redundant constraints in the halfspace representation
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        n: dimension of the polytope
        tol: tolerance
        
    Returns:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        empty: true/false whether polytope is empty
        minHRep: minimal representation obtained
    """
    minHRep = False
    
    # Remove all constraints where the vector is all-zero
    A, b, Ae, be, empty = priv_compact_zeros(A, b, Ae, be, tol)
    if empty:
        return A, b, Ae, be, empty, minHRep
    
    # Equality constraints can only be redundant if they are aligned
    A, b, Ae, be = priv_compact_toEquality(A, b, Ae, be, tol)
    Ae, be, empty = priv_compact_alignedEq(Ae, be, tol)
    if empty:
        return A, b, Ae, be, empty, minHRep
    
    A, b = priv_compact_alignedIneq(A, b, tol)
    
    # Special algorithms for 1D and 2D, general method for nD
    if n == 1:
        A, b, Ae, be, empty = priv_compact_1D(A, b, Ae, be, tol)
    elif n == 2:
        A, b, Ae, be, empty = priv_compact_2D(A, b, Ae, be, tol)
    else:
        A, b, empty = priv_compact_nD(A, b, Ae, be, n, tol)
    
    minHRep = True
    
    return A, b, Ae, be, empty, minHRep 