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
        argsIn_i = sp.symbols(f'{varName}1:{f_in[i]+1}', real=True)
        if f_in[i] == 1:
            argsIn.append(argsIn_i)
        else:
            argsIn.append(sp.Matrix([sp.Symbol(f'{varName}{j+1}', real=True) for j in range(f_in[i])]))
    
    # MATLAB: f_sym = f(argsIn{:});
    f_sym = f(*argsIn)
    # MATLAB: g_sym = g(argsIn{:});
    g_sym = g(*argsIn)
    
    # check symbolic expressions for equality
    # MATLAB: res = isequal(f_sym,g_sym);
    # In sympy, we use equals() for symbolic equality
    try:
        if isinstance(f_sym, (list, tuple, np.ndarray)) and isinstance(g_sym, (list, tuple, np.ndarray)):
            if len(f_sym) != len(g_sym):
                return False
            # Compare element-wise
            for fi, gi in zip(f_sym, g_sym):
                if hasattr(fi, 'equals') and hasattr(gi, 'equals'):
                    if not fi.equals(gi):
                        return False
                elif fi != gi:
                    return False
            return True
        elif hasattr(f_sym, 'equals') and hasattr(g_sym, 'equals'):
            return f_sym.equals(g_sym)
        else:
            return f_sym == g_sym
    except:
        # If comparison fails, assume not equal
        return False

