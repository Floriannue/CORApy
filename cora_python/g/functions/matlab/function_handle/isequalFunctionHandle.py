"""
isequalFunctionHandle - checks if two function handles are equal

Syntax:
    res = isequalFunctionHandle(f,g)

Inputs:
    f - function handle
    g - function handle

Outputs:
    res - true/false

Example:
    f = lambda x, u: [x[0] - u[0], x[0]*x[1]]
    g = lambda x, u: [x[0] - u[0], x[1]*x[0]]
    isequalFunctionHandle(f,g);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       07-October-2024 
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import sympy as sp
import numpy as np
from typing import Callable
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.function_handle.inputArgsLength import inputArgsLength


def isequalFunctionHandle(f: Callable, g: Callable) -> bool:
    """
    Checks if two function handles are equal
    
    Args:
        f: function handle
        g: function handle
        
    Returns:
        res: True if function handles are equal, False otherwise
    """
    
    # ensure that we have two function handles
    # MATLAB: narginchk(2,2);
    # MATLAB: inputArgsCheck({{f,'att','function_handle'}; {g,'att','function_handle'}});
    # In Python, we check if they are callable
    if not callable(f):
        raise TypeError("f must be a function handle (callable)")
    if not callable(g):
        raise TypeError("g must be a function handle (callable)")
    
    # get the number of input/output arguments for each function handle
    # MATLAB: [f_in,f_out] = inputArgsLength(f);
    f_in, f_out = inputArgsLength(f)
    # MATLAB: [g_in,g_out] = inputArgsLength(g);
    g_in, g_out = inputArgsLength(g)
    
    # number of input arguments and their size must match
    # MATLAB: if any(size(f_in) ~= size(g_in)) || any(f_in ~= g_in)
    # Note: MATLAB allows different input dimensions if one has dummy inputs
    # We need to use the maximum dimensions for evaluation
    if len(f_in) != len(g_in):
        return False
    
    # Check if dimensions match (allowing for dummy inputs where one is 0)
    # If dimensions don't match and neither is 0, they're not equal
    dims_match = True
    max_dims = []
    for i in range(len(f_in)):
        if f_in[i] != g_in[i]:
            # If one is 0, it's a dummy input - use the non-zero one
            if f_in[i] == 0:
                max_dims.append(g_in[i])
            elif g_in[i] == 0:
                max_dims.append(f_in[i])
            else:
                # Both non-zero and different - not equal
                return False
        else:
            max_dims.append(f_in[i])
    
    # Use max_dims for creating symbolic variables (handles dummy inputs)
    eval_dims = max_dims
    
    # number of output arguments must match
    # MATLAB: if f_out ~= g_out
    if f_out != g_out:
        return False
    
    # insert symbolic variables into the functions
    # MATLAB: numArgsIn = numel(f_in);
    numArgsIn = len(eval_dims)
    # MATLAB: argsIn = cell(numArgsIn,1);
    argsIn = []
    # MATLAB: for i=1:numArgsIn
    for i in range(numArgsIn):
        # MATLAB: varName = ['argsIn_' num2str(i) '_'];
        varName = f'argsIn_{i}_'
        # MATLAB: argsIn{i} = sym(varName,[f_in(i),1],'real');
        # Create column vector of symbolic variables using eval_dims (handles dummy inputs)
        # Always create as numpy array to allow indexing (even for dimension 1)
        # This matches how inputArgsLength creates variables
        if eval_dims[i] == 1:
            # Single element array (can still be indexed)
            argsIn.append(np.array([sp.Symbol(f'{varName}1', real=True)]))
        else:
            # Column vector as numpy array
            argsIn.append(np.array([sp.Symbol(f'{varName}{j+1}', real=True) for j in range(eval_dims[i])]))
    
    # MATLAB: f_sym = f(argsIn{:});
    f_sym = f(*argsIn)
    # MATLAB: g_sym = g(argsIn{:});
    g_sym = g(*argsIn)
    
    # check symbolic expressions for equality
    # MATLAB: res = isequal(f_sym,g_sym);
    # MATLAB's isequal compares symbolic expressions element-wise
    # MATLAB does NOT catch exceptions - errors propagate
    # Convert numpy arrays to sympy matrices if needed
    # Handle numpy arrays containing sympy symbols
    # MATLAB: isequal works directly on symbolic arrays/matrices
    if isinstance(f_sym, np.ndarray):
        # Convert numpy array to sympy Matrix
        # Handle arrays containing sympy symbols by converting element-wise
        if f_sym.size == 0:
            f_sym = sp.Matrix([])
        else:
            # Try direct conversion first (works for numeric arrays)
            try:
                f_sym = sp.Matrix(f_sym)
            except:
                # If that fails, convert to list first (needed for arrays with sympy objects)
                try:
                    f_sym = sp.Matrix(f_sym.tolist())
                except:
                    # Last resort: flatten and reshape
                    flat = f_sym.flatten()
                    f_sym = sp.Matrix([flat[i] for i in range(len(flat))]).reshape(f_sym.shape[0], f_sym.shape[1] if f_sym.ndim > 1 else 1)
    if isinstance(g_sym, np.ndarray):
        if g_sym.size == 0:
            g_sym = sp.Matrix([])
        else:
            try:
                g_sym = sp.Matrix(g_sym)
            except:
                try:
                    g_sym = sp.Matrix(g_sym.tolist())
                except:
                    flat = g_sym.flatten()
                    g_sym = sp.Matrix([flat[i] for i in range(len(flat))]).reshape(g_sym.shape[0], g_sym.shape[1] if g_sym.ndim > 1 else 1)
    
    # Handle lists/tuples - convert to sympy Matrix
    if isinstance(f_sym, (list, tuple)):
        f_sym = sp.Matrix(f_sym)
    if isinstance(g_sym, (list, tuple)):
        g_sym = sp.Matrix(g_sym)
    
    # Both should be sympy objects now
    # Use simplify and equals for proper symbolic comparison
    # MATLAB's isequal for symbolic does element-wise comparison
    if isinstance(f_sym, sp.Matrix) and isinstance(g_sym, sp.Matrix):
        if f_sym.shape != g_sym.shape:
            return False
        # Compare element-wise using simplify and check if difference is zero
        # MATLAB: sym_diff = simplify(f1-f2); idx_diff_zero = logical(sym_diff == 0); if all(idx_diff_zero)
        # MATLAB's isequal for symbolic matrices does element-wise comparison using == 0
        diff = sp.simplify(f_sym - g_sym)
        # Check if all elements are zero using == 0 (MATLAB's approach)
        # MATLAB: idx_diff_zero = logical(sym_diff == 0); if all(idx_diff_zero)
        for i in range(diff.rows):
            for j in range(diff.cols):
                elem = sp.simplify(diff[i, j])
                # MATLAB uses == 0 directly on simplified expression
                # This works for sympy too - == 0 returns True/False for zero expressions
                if not (elem == 0):
                    return False
        return True
    elif isinstance(f_sym, (sp.Expr, sp.Symbol, sp.Basic)) and isinstance(g_sym, (sp.Expr, sp.Symbol, sp.Basic)):
        # Scalar symbolic expressions
        diff = sp.simplify(f_sym - g_sym)
        return diff == 0
    else:
        # Fallback to direct comparison (MATLAB would throw error for unsupported types)
        return f_sym == g_sym

