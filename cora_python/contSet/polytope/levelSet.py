"""
levelSet - converts polytope object to an equivalent levelSet object

Syntax:
   ls = levelSet(P)

Inputs:
   P - polytope object

Outputs:
   ls - levelSet object

Authors:       Maximilian Perschl (MATLAB)
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, List
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.levelSet.levelSet import LevelSet

# Import symbolic computation capabilities
try:
    import sympy as sp
    from sympy import symbols
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

if TYPE_CHECKING:
    from .polytope import Polytope

def levelSet(P: 'Polytope') -> 'LevelSet':
    """
    Converts a polytope object to an equivalent levelSet object.

    Args:
        P: Polytope object.

    Returns:
        LevelSet object.
    """
    # Compute halfspace representation if only vertices given
    P.constraints()

    # Check for equality constraints (MATLAB lines 33-35)
    if P.Ae.size > 0:
        raise CORAerror('CORA:notSupported','Equality constraints not supported for levelSet conversion.')

    # Read out variables, A, and b
    n = P.dim()
    
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not installed. Cannot perform symbolic operations for LevelSet.")

    # MATLAB: vars = sym('x',[dim(P), 1]);
    # This creates a column vector of symbolic variables x0, x1, ..., x(n-1)
    vars_sym = symbols([f'x{i}' for i in range(n)]) # Creates tuple, need list for consistency
    # Convert to a list of symbols if not already, and potentially reshape to column conceptually
    if n == 1:
        vars_list = [vars_sym]
    else:
        vars_list = list(vars_sym)

    A = P.A
    b = P.b

    # Init symbolic equations (MATLAB: eq = A*vars - b;)
    # Need to perform matrix multiplication of sympy matrix with sympy symbols
    # Convert A and b to sympy matrices/vectors first if they are numpy arrays
    # It's better to build the expression component-wise to avoid complex matrix setup for sympy

    eq_list = []
    for i in range(A.shape[0]):
        row_expr = 0
        for j in range(A.shape[1]):
            row_expr += A[i, j] * vars_list[j]
        eq_list.append(row_expr - sp.Float(b[i, 0])) # Ensure b[i,0] is a SymPy Float
    
    # If there's only one equation, make it a single sympy expression, not a list of one
    if len(eq_list) == 1:
        eq_sym = eq_list[0]
    elif len(eq_list) > 1:
        # For multiple inequalities, it's typically a list of expressions in SymPy for LevelSet
        eq_sym = eq_list
    else:
        # Handle case with no inequality constraints (e.g., empty polytope)
        eq_sym = sp.Integer(0) # Represent an always-true constraint if no inequalities
        vars_list = [symbols('x0')] # Need at least one symbol for dim calculation
        if n > 0 and A.shape[0] == 0: # If dimension is positive but no constraints
            # Create dummy symbols based on dimension if no constraints
            vars_list = symbols([f'x{i}' for i in range(n)])
            if n == 1: vars_list = [vars_list]
            else: vars_list = list(vars_list)


    # Stack <= comparison operator times the number of inequalities
    # MATLAB: compOps = repmat({'<='},length(P.b_.val),1);
    if isinstance(eq_sym, list):
        compOps = ['<='] * len(eq_sym)
    else: # Single equation
        compOps = '<='

    # Init resulting level set
    ls = LevelSet(eq_sym, vars_list, compOps)

    return ls
