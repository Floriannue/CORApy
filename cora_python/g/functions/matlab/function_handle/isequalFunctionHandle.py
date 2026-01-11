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
    if len(f_in) != len(g_in) or not np.array_equal(f_in, g_in):
        return False
    
    # number of output arguments must match
    # MATLAB: if f_out ~= g_out
    if f_out != g_out:
        return False
    
    # insert symbolic variables into the functions
    # MATLAB: numArgsIn = numel(f_in);
    numArgsIn = len(f_in)
    # MATLAB: argsIn = cell(numArgsIn,1);
    argsIn = []
    # MATLAB: for i=1:numArgsIn
    for i in range(numArgsIn):
        # MATLAB: varName = ['argsIn_' num2str(i) '_'];
        varName = f'argsIn_{i}_'
        # MATLAB: argsIn{i} = sym(varName,[f_in(i),1],'real');
        # Create column vector of symbolic variables
        if f_in[i] == 1:
            # Single symbol
            argsIn.append(sp.Symbol(f'{varName}1', real=True))
        else:
            # Column vector (Matrix)
            argsIn.append(sp.Matrix([sp.Symbol(f'{varName}{j+1}', real=True) for j in range(f_in[i])]))
    
    # MATLAB: f_sym = f(argsIn{:});
    f_sym = f(*argsIn)
    # MATLAB: g_sym = g(argsIn{:});
    g_sym = g(*argsIn)
    
    # check symbolic expressions for equality
    # MATLAB: res = isequal(f_sym,g_sym);
    # MATLAB's isequal compares symbolic expressions element-wise
    try:
        # Convert numpy arrays to sympy matrices if needed
        if isinstance(f_sym, np.ndarray):
            f_sym = sp.Matrix(f_sym)
        if isinstance(g_sym, np.ndarray):
            g_sym = sp.Matrix(g_sym)
        
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
            # MATLAB's isequal for symbolic matrices does element-wise comparison
            diff = sp.simplify(f_sym - g_sym)
            # Check if all elements are zero (use is_zero for better symbolic comparison)
            for i in range(diff.rows):
                for j in range(diff.cols):
                    elem = sp.simplify(diff[i, j])
                    # Check if element is zero
                    if not (elem == 0 or (hasattr(elem, 'is_zero') and elem.is_zero())):
                        return False
            return True
        elif isinstance(f_sym, (sp.Expr, sp.Symbol)) and isinstance(g_sym, (sp.Expr, sp.Symbol)):
            # Scalar symbolic expressions
            diff = sp.simplify(f_sym - g_sym)
            return diff == 0
        else:
            # Fallback to direct comparison
            return f_sym == g_sym
    except Exception as e:
        # If comparison fails, assume not equal
        return False

