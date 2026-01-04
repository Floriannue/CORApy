"""
derive - compute customizable derivatives of nonlinear functions and save
   them as .py files; the number of input arguments is the length of the
   list for the provided set of variables (this can also be read
   automatically from the provided function handle), the number of output
   arguments is the derived tensor with respect to each variable in the
   set of variables

Syntax:
    derive('FunctionHandle',f)
    derive('SymbolicFunction',f_sym)
    ...

Inputs:
    Name-Value pairs (all options, arbitrary order):
       <'FunctionHandle',f> - function handle
       <'SymbolicFunction',f_sym> - symbolic function (nD array of sympy symbols)
       <'Vars',vars> - set of variables w.r.t which function is derived
       <'VarNamesIn',varNamesIn> - variable names for input arguments
       <'VarNamesOut',varNamesOut> - variable names for output arguments
       <'Path',fpath> - path where to save generated .py file
       <'FileName',fname> - name of generated file
       <'Verbose',verbose> - true/false for verbose output
       <'IntervalArithmetic',isInt> - true/false whether derivative is to
           be evaluated using interval arithmetic
       (extend to .lagrangeRem, parametric, ...)

Outputs:
    symDerivative - sympy object containing the symbolic derivative
    handle - handle to the generated .py file

Example:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: writeMatrixFile

Authors:       Mark Wetzlinger
Written:       12-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import sympy as sp
import numpy as np
import inspect
from typing import Any, List, Optional, Tuple, Callable, Union
from cora_python.g.functions.matlab.validate.check.checkNameValuePairs import checkNameValuePairs
from cora_python.g.functions.matlab.validate.preprocessing.readNameValuePair import readNameValuePair
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.verbose.write.writeMatrixFile import writeMatrixFile
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning
from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.g.functions.matlab.function_handle.inputArgsLength import inputArgsLength
import os


def derive(*varargin) -> Tuple[Any, Optional[Callable]]:
    """
    Compute customizable derivatives of nonlinear functions and save them as .py files
    
    Args:
        *varargin: Name-value pairs:
            'FunctionHandle': function handle
            'SymbolicFunction': symbolic function (sympy expression)
            'Vars': list of variables
            'VarNamesIn': list of input variable names
            'VarNamesOut': list of output variable names
            'Path': path where to save generated file
            'FileName': name of generated file
            'Verbose': bool for verbose output
            'IntervalArithmetic': bool whether to use interval arithmetic
            
    Returns:
        symDerivative: sympy object containing the symbolic derivative
        handle: callable handle to the generated .py file (or None)
    """
    
    # name-value pairs -> number of input arguments is always a multiple of 2
    if len(varargin) % 2 != 0:
        raise CORAerror('CORA:evenNumberInputArgs')
    
    # read input arguments
    NVpairs = list(varargin)
    # check list of name-value pairs
    checkNameValuePairs(NVpairs, ['FunctionHandle', 'SymbolicFunction',
                                   'Vars', 'VarNamesIn', 'VarNamesOut', 'Path', 
                                   'FileName', 'IntervalArithmetic', 'Verbose'])
    
    # function handle given?
    def is_function_handle(x):
        return callable(x) and not isinstance(x, (sp.Basic, sp.Matrix))
    
    NVpairs, f = readNameValuePair(NVpairs, 'FunctionHandle',
                                    is_function_handle, lambda: None)
    
    # symbolic function given?
    def is_sympy_expr(x):
        return isinstance(x, (sp.Basic, sp.Matrix)) or (hasattr(x, '__class__') and 
                                                         'sympy' in str(type(x)))
    
    NVpairs, f_sym = readNameValuePair(NVpairs, 'SymbolicFunction',
                                        is_sympy_expr, sp.Matrix([]))
    
    # path for storage given?
    defaultPath = os.path.join(CORAROOT(), 'models', 'auxiliary')
    NVpairs, fpath = readNameValuePair(NVpairs, 'Path',
                                        lambda x: isinstance(x, str), defaultPath)
    
    # name of system given?
    NVpairs, fname = readNameValuePair(NVpairs, 'FileName',
                                        lambda x: isinstance(x, str), 'derivative')
    
    # interval arithmetic true/false?
    NVpairs, intervalOn = readNameValuePair(NVpairs, 'IntervalArithmetic',
                                            lambda x: isinstance(x, bool), False)
    
    # verbose output?
    NVpairs, verbose = readNameValuePair(NVpairs, 'Verbose',
                                         lambda x: isinstance(x, bool), False)
    
    # symbolic variables, names for input/output arguments given?
    # ...their default values are a bit more complicated
    NVpairs, vars = readNameValuePair(NVpairs, 'Vars')
    NVpairs, varNamesIn = readNameValuePair(NVpairs, 'VarNamesIn')
    NVpairs, varNamesOut = readNameValuePair(NVpairs, 'VarNamesOut')
    
    # set all remaining values
    vars, varNamesIn, varNamesOut = aux_setDefaultValues(vars, varNamesIn, varNamesOut, f)
    
    # check values
    # Skip f_sym validation if it's empty (will be populated later if needed)
    check_list = [
        [f, 'att', 'function_handle', 'scalar'],
        [vars, 'att', 'cell'],
        [varNamesIn, 'att', 'cell'],
        [varNamesOut, 'att', 'cell'],
        [fpath, 'att', 'char'],
        [fname, 'att', 'char'],
        [intervalOn, 'att', 'logical', 'scalar'],
        [verbose, 'att', 'logical', 'scalar']
    ]
    # Only validate f_sym if it's not empty
    if f_sym is not None and not (isinstance(f_sym, sp.Matrix) and f_sym.shape == (0, 0)):
        check_list.insert(1, [f_sym, 'att', 'sym'])
    inputArgsCheck(check_list)
    
    # evaluate function handle if necessary, otherwise use symbolic function
    f_sym = aux_getSymbolicFunction(f_sym, f, vars)
    
    # derive using sympy jacobian
    symDerivative = aux_derive(f_sym, vars)
    
    # skip file generation if path is 'none'
    if fpath == 'none':
        handle = None
        return symDerivative, handle
    
    # substitute x1 -> xL1R, etc. so that we can write the file correctly
    numVars = len(vars)
    vars_LR = []
    for i in range(numVars):
        var_name = varNamesIn[i]
        var_dim = len(vars[i]) if hasattr(vars[i], '__len__') else 1
        # Create symbols like xL1R, xL2R, etc.
        var_symbols = [sp.Symbol(f'{var_name}L{j}R', real=True) for j in range(1, var_dim + 1)]
        vars_LR.append(var_symbols)
    
    # Flatten old and new variables for substitution
    oldVars = []
    newVars = []
    for var_list in vars:
        if isinstance(var_list, (list, tuple)):
            oldVars.extend(var_list)
        else:
            oldVars.append(var_list)
    for var_list in vars_LR:
        if isinstance(var_list, (list, tuple)):
            newVars.extend(var_list)
        else:
            newVars.append(var_list)
    
    # Perform substitution
    symDerivative_LR = []
    for M in symDerivative:
        if isinstance(M, (sp.Basic, sp.Matrix)):
            # Create substitution dictionary
            subs_dict = {old: new for old, new in zip(oldVars, newVars)}
            symDerivative_LR.append(M.subs(subs_dict))
        else:
            symDerivative_LR.append(M)
    
    # generate file
    handle = writeMatrixFile(symDerivative_LR, fpath, fname,
                             'VarNamesIn', varNamesIn, 'VarNamesOut', varNamesOut,
                             'BracketSubs', True, 'IntervalArithmetic', intervalOn)
    
    return symDerivative, handle


# Auxiliary functions -----------------------------------------------------

def aux_derive(f_sym: Any, vars: List[Any]) -> List[Any]:
    """
    Compute jacobian derivatives using sympy
    
    Args:
        f_sym: symbolic function (sympy expression or matrix)
        vars: list of variables to derive with respect to
        
    Returns:
        symDerivative: list of derivatives (one per variable)
    """
    
    # check size of symbolic expression to call 'jacobian' correctly
    if isinstance(f_sym, sp.Matrix):
        sz = f_sym.shape
        f_sym_flat = f_sym
    elif isinstance(f_sym, (list, tuple, np.ndarray)):
        f_sym = np.asarray(f_sym)
        sz = f_sym.shape
        # Convert to sympy Matrix
        if f_sym.ndim == 1:
            f_sym_flat = sp.Matrix(f_sym.tolist())
        else:
            f_sym_flat = sp.Matrix(f_sym.tolist())
    else:
        # Assume it's a sympy expression
        sz = (1, 1) if not hasattr(f_sym, 'shape') else f_sym.shape
        f_sym_flat = sp.Matrix([f_sym]) if not isinstance(f_sym, sp.Matrix) else f_sym
    
    # number of variables
    numVars = len(vars)
    symDerivative = [None] * numVars
    
    # support only up until 3D for now... nD version probably requires
    # recursion (no time one week before v2025)
    if len(sz) > 3:
        raise CORAerror('CORA:notSupported',
                       'The function derive currently only supports max. 3D sym matrices.')
    
    # Convert vars to sympy symbols if needed
    vars_sympy = []
    for var_list in vars:
        if isinstance(var_list, (list, tuple)):
            vars_sympy.append([sp.Symbol(str(v), real=True) if not isinstance(v, sp.Basic) else v 
                               for v in var_list])
        else:
            vars_sympy.append([sp.Symbol(str(var_list), real=True) if not isinstance(var_list, sp.Basic) 
                               else var_list])
    
    if len(sz) == 1 or (len(sz) == 2 and (sz[0] == 1 or sz[1] == 1)):
        # vectors are ok as they are... reshape has no effect on 'jacobian' but
        # might make it more obvious what is happening
        if isinstance(f_sym_flat, sp.Matrix):
            f_sym_vec = f_sym_flat.reshape(len(f_sym_flat), 1) if f_sym_flat.shape[1] != 1 else f_sym_flat
        else:
            f_sym_vec = sp.Matrix([f_sym_flat])
        
        # Compute jacobian for each variable
        for i in range(numVars):
            var_vec = sp.Matrix(vars_sympy[i])
            symDerivative[i] = f_sym_vec.jacobian(var_vec)
    
    elif len(sz) == 2:
        # 2D array (no vector) -> 3D tensor
        for i in range(numVars):
            var_vec = sp.Matrix(vars_sympy[i])
            # For each row, compute jacobian
            jacobian_list = []
            for j in range(sz[0]):
                row_expr = f_sym_flat.row(j)
                jacobian_list.append(row_expr.jacobian(var_vec))
            # Stack results (this creates a 3D structure)
            symDerivative[i] = sp.Matrix(jacobian_list)
    
    elif len(sz) == 3:
        # 3D array -> 4D tensor
        for i in range(numVars):
            var_vec = sp.Matrix(vars_sympy[i])
            # For each slice, compute jacobian
            jacobian_list = []
            for k in range(sz[0]):
                slice_list = []
                for l in range(sz[1]):
                    # Extract slice (k, l, :)
                    slice_expr = sp.Matrix([f_sym_flat[k, l, m] for m in range(sz[2])])
                    slice_list.append(slice_expr.jacobian(var_vec))
                jacobian_list.append(slice_list)
            # This creates a 4D structure (nested matrices)
            symDerivative[i] = jacobian_list
    
    return symDerivative


def aux_getSymbolicFunction(f_sym: Any, f: Optional[Callable], vars: List[Any]) -> Any:
    """
    Get symbolic function from function handle or use provided symbolic function
    
    Args:
        f_sym: symbolic function (may be empty)
        f: function handle (may be None)
        vars: list of variables
        
    Returns:
        f_sym: symbolic function
    """
    
    # prefer symbolic function over function handle (warning if both given)
    # note: we require the string because the default value (f = lambda: []) does
    # not return true when inserted into isempty)
    f_text = ""
    if f is not None:
        try:
            # Try to get function source code or name
            if hasattr(f, '__name__'):
                f_text = f.__name__
            elif hasattr(f, '__code__'):
                f_text = str(f.__code__)
            else:
                f_text = str(f)
            # Remove common function handle characters
            f_text = f_text.replace('@', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        except:
            f_text = ""
    
    if f_text and f_sym is not None and not (isinstance(f_sym, sp.Matrix) and f_sym.shape == (0, 0)):
        CORAwarning("CORA:global",
                   "Either provide function handle or symbolic function.\nSymbolic function used.")
    
    # evaluate symbolic function
    if f_sym is None or (isinstance(f_sym, sp.Matrix) and f_sym.shape == (0, 0)):
        if f is None:
            raise CORAerror('CORA:specialError',
                          'Either function handle or symbolic function must be provided.')
        try:
            # Evaluate function with symbolic variables
            # Flatten vars for function call
            vars_flat = []
            for var_list in vars:
                if isinstance(var_list, (list, tuple)):
                    vars_flat.extend(var_list)
                else:
                    vars_flat.append(var_list)
            f_sym = f(*vars_flat)
            # Convert to sympy if needed
            if not isinstance(f_sym, (sp.Basic, sp.Matrix)):
                f_sym = sp.Matrix(f_sym) if isinstance(f_sym, (list, tuple, np.ndarray)) else sp.sympify(f_sym)
        except Exception as e:
            raise CORAerror('CORA:specialError',
                          'Variables do not match given function handle.') from e
    
    return f_sym


def aux_setDefaultValues(vars: Optional[List[Any]], varNamesIn: Optional[List[str]], 
                         varNamesOut: Optional[List[str]], f: Optional[Callable]) -> Tuple[List[Any], List[str], List[str]]:
    """
    Set default values for variables and names
    
    Args:
        vars: list of variables (may be None/empty)
        varNamesIn: list of input variable names (may be None/empty)
        varNamesOut: list of output variable names (may be None/empty)
        f: function handle (used if vars is empty)
        
    Returns:
        vars: list of variables
        varNamesIn: list of input variable names
        varNamesOut: list of output variable names
    """
    
    # if vars not given, use inputArgsLength to generate variables of
    # appropriate length: example f(x,u) with x in R^3, u in R^2 yields
    #    vars[0] = [in1_1, in1_2, in1_3]
    #    vars[1] = [in2_1, in2_2]
    # (we use 'in' analogously to matlabFunction)
    if vars is None or (isinstance(vars, list) and len(vars) == 0):
        if f is None:
            raise CORAerror('CORA:specialError',
                          'Either vars or function handle must be provided.')
        count, _ = inputArgsLength(f)
        numInputArgs = len(count)
        maxInputArgLength = max(count) if count else 1
        
        # Create symbolic variables
        vars_raw = []
        for i in range(numInputArgs):
            var_list = []
            for j in range(count[i]):
                var_list.append(sp.Symbol(f'in{i+1}_{j+1}', real=True))
            vars_raw.append(var_list)
        
        vars = vars_raw
    
    # default input arguments
    if varNamesIn is None or (isinstance(varNamesIn, list) and len(varNamesIn) == 0):
        varNamesIn = [f'in{i+1}' for i in range(len(vars))]
    
    # default: variable names out1, out2, ...
    if varNamesOut is None or (isinstance(varNamesOut, list) and len(varNamesOut) == 0):
        varNamesOut = [f'out{i+1}' for i in range(len(vars))]
    
    return vars, varNamesIn, varNamesOut

