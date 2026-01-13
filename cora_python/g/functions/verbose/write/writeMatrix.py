"""
writeMatrix - writes the content of a symbolic matrix with a custom
   variable name to a file; for aesthetic reasons, we differentiate
   between 2D, 3D, and nD cases

Syntax:
    writeMatrix(fid,M,varName)
    writeMatrix(fid,M,varName,...)

Inputs:
    fid - identifier of the file to which the matrix is written
    M - symbolic nD matrix
    varName - variable name
    Name-Value pairs (all optional, arbitrary order):
       <'BracketSubs',bracketSubsOn> - true/false whether bracketSubs
           should be called
       <'Sparse',sparseOn> - true/false whether matrix should be sparse
       <'IntervalArithmetic',intervalOn> - true/false whether output
           matrix should be converted to an interval

Outputs:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: matlabFunction, writeMatrixFile, bracketSubs

Authors:       Mark Wetzlinger
Written:       13-October-2024
Last update:   --- 
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import sympy as sp
import numpy as np
from typing import Any, TextIO
from cora_python.g.functions.matlab.validate.preprocessing.readNameValuePair import readNameValuePair
from cora_python.g.functions.matlab.validate.check.checkNameValuePairs import checkNameValuePairs
from cora_python.g.functions.matlab.string.bracketSubs import bracketSubs
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def writeMatrix(fid: TextIO, M: Any, varName: str, *varargin) -> None:
    """
    Writes the content of a symbolic matrix with a custom variable name to a file
    
    Args:
        fid: file identifier (file handle) to which the matrix is written
        M: symbolic nD matrix (sympy Matrix or array)
        varName: variable name
        *varargin: optional name-value pairs:
            'BracketSubs': bool - whether bracketSubs should be called
            'Sparse': bool - whether matrix should be sparse
            'IntervalArithmetic': bool - whether output matrix should be converted to an interval
    """
    
    # MATLAB: narginchk(3,Inf);
    # MATLAB: if mod(nargin,2) ~= 1
    if len(varargin) % 2 != 0:
        raise CORAerror('CORA:oddNumberInputArgs')
    
    # read out size of matrix
    # MATLAB: sz = size(M);
    if isinstance(M, sp.Matrix):
        sz = M.shape
    elif isinstance(M, np.ndarray):
        sz = M.shape
    elif hasattr(M, 'shape'):
        sz = M.shape
    else:
        # Try to convert to sympy Matrix
        M = sp.Matrix(M)
        sz = M.shape
    
    # read input arguments
    # MATLAB: NVpairs = varargin(1:end);
    NVpairs = list(varargin)
    # check list of name-value pairs
    # MATLAB: checkNameValuePairs(NVpairs,{'BracketSubs','Sparse','IntervalArithmetic'});
    checkNameValuePairs(NVpairs, ['BracketSubs', 'Sparse', 'IntervalArithmetic'])
    
    # function handle given?
    # MATLAB: [NVpairs,bracketSubsOn] = readNameValuePair(NVpairs,'BracketSubs',@islogical,false);
    NVpairs, bracketSubsOn = readNameValuePair(NVpairs, 'BracketSubs', lambda x: isinstance(x, bool), False)
    
    # symbolic function given? (default value determined by size)
    # MATLAB: [NVpairs,sparseOn] = readNameValuePair(NVpairs,'Sparse',@islogical,numel(sz)>=3);
    NVpairs, sparseOn = readNameValuePair(NVpairs, 'Sparse', lambda x: isinstance(x, bool), len(sz) >= 3)
    
    # interval conversion desired?
    # MATLAB: [NVpairs,intervalOn] = readNameValuePair(NVpairs,'IntervalArithmetic',@islogical,false);
    NVpairs, intervalOn = readNameValuePair(NVpairs, 'IntervalArithmetic', lambda x: isinstance(x, bool), False)
    
    # different formatting for different dimensions of the matrix
    # MATLAB: if numel(sz) == 2
    if len(sz) == 2:
        # MATLAB: aux_write2D(fid,M,varName,bracketSubsOn,sparseOn,intervalOn);
        aux_write2D(fid, M, varName, bracketSubsOn, sparseOn, intervalOn)
    # MATLAB: elseif numel(sz) == 3
    elif len(sz) == 3:
        # MATLAB: aux_write3D(fid,M,varName,bracketSubsOn,sparseOn,intervalOn);
        aux_write3D(fid, M, varName, bracketSubsOn, sparseOn, intervalOn)
    else:
        # MATLAB: aux_writenD(fid,M,varName,bracketSubsOn,sparseOn,intervalOn);
        aux_writenD(fid, M, varName, bracketSubsOn, sparseOn, intervalOn)


# Auxiliary functions -----------------------------------------------------

def aux_write2D(fid: TextIO, M: Any, varName: str, bracketSubsOn: bool, 
                sparseOn: bool, intervalOn: bool) -> None:
    """
    1D, 2D matrix writing
    """
    
    # Convert sympy Matrix to Python code using sympy's code generation
    # MATLAB: if bracketSubsOn
    if bracketSubsOn:
        # MATLAB: M_char = bracketSubs(char(M));
        # For bracket substitution, we need to convert the matrix to string first
        # then apply bracketSubs, then convert MATLAB indexing to Python
        M_str = _matrix_to_python_code(M)
        M_char = bracketSubs(M_str)
        # Convert MATLAB-style indexing x(1) to Python indexing x[0]
        M_char = _convert_matlab_indexing_to_python(M_char)
    else:
        # MATLAB: M_char = char(M);
        M_char = _matrix_to_python_code(M)
        # Still need to convert indexing even without bracketSubs
        M_char = _convert_matlab_indexing_to_python(M_char)
    
    # additional text: sparse, then interval
    # MATLAB: if sparseOn
    if sparseOn:
        # MATLAB: str = sprintf('sparse(%s)', str);
        M_char = f'scipy.sparse.csr_matrix({M_char})'
    
    # MATLAB: if intervalOn
    if intervalOn:
        # MATLAB: str = sprintf('interval(%s)', str);
        M_char = f'Interval({M_char})'
    
    # write to file
    # MATLAB: fprintf(fid, '%s = %s;\n\n', varName, str);
    fid.write(f'    {varName} = {M_char}\n\n')


def aux_write3D(fid: TextIO, M: Any, varName: str, bracketSubsOn: bool,
                sparseOn: bool, intervalOn: bool) -> None:
    """
    3D: read out content of each page
    """
    
    # MATLAB: sz = size(M);
    if isinstance(M, sp.Matrix):
        sz = M.shape
    elif isinstance(M, np.ndarray):
        sz = M.shape
    else:
        sz = M.shape
    
    # init the matrix by zeros, enclose by sparse/interval if necessary
    # MATLAB: initStr = sprintf('zeros(%i,%i,%i)', sz);
    initStr = f'np.zeros(({sz[0]}, {sz[1]}, {sz[2]}))'
    
    # MATLAB: if sparseOn
    if sparseOn:
        # MATLAB: initStr = sprintf('sparse(%s)', initStr);
        initStr = f'scipy.sparse.csr_matrix({initStr})'
    
    # MATLAB: if intervalOn
    if intervalOn:
        # MATLAB: initStr = sprintf('interval(%s)', initStr);
        initStr = f'Interval({initStr})'
    
    # MATLAB: fprintf(fid, '%s = %s;\n\n', varName, initStr);
    fid.write(f'    {varName} = {initStr}\n\n')
    
    # insert value if page is non-zero
    # MATLAB: for k=1:sz(3)
    for k in range(sz[2]):
        # skip line if entries are all-zero (already zero via initialization)
        # MATLAB: if all(M(:,:,k) == zeros(sz(1),sz(2)),'all')
        M_slice = M[:, :, k] if isinstance(M, np.ndarray) else M[:, :, k]
        zero_matrix = np.zeros((sz[0], sz[1]))
        
        # Check if slice is all zeros
        if isinstance(M_slice, sp.Matrix):
            is_all_zero = all(M_slice[i, j] == 0 for i in range(sz[0]) for j in range(sz[1]))
        else:
            is_all_zero = np.allclose(M_slice, zero_matrix)
        
        if is_all_zero:
            # MATLAB: continue
            continue
        
        # MATLAB: if bracketSubsOn
        if bracketSubsOn:
            # MATLAB: M_char = bracketSubs(char(M(:,:,k)));
            M_str = str(M_slice)
            M_char = bracketSubs(M_str)
        else:
            # MATLAB: M_char = char(M(:,:,k));
            M_char = str(M_slice)
        
        # Convert to Python format
        str_py = _matrix_to_python_code(M_slice)
        
        # MATLAB: fprintf(fid, '%s(:,:,%i) = %s;\n\n', varName, k, str);
        # Python uses 0-based indexing, but we write k+1 for MATLAB compatibility in generated code
        # Actually, we should use k (0-based) for Python
        fid.write(f'    {varName}[:, :, {k}] = np.array({str_py})\n\n')


def aux_writenD(fid: TextIO, M: Any, varName: str, bracketSubsOn: bool,
                sparseOn: bool, intervalOn: bool) -> None:
    """
    nD: use reshape
    """
    
    # sparse and interval currently not supported
    # MATLAB: if sparseOn || intervalOn
    if sparseOn or intervalOn:
        # MATLAB: throw(CORAerror('CORA:notSupported',...))
        raise CORAerror('CORA:notSupported',
                       'Sparsity and conversion to interval currently not supported for matrices larger than 3D.')
    
    # MATLAB: if bracketSubsOn
    if bracketSubsOn:
        # MATLAB: M_char = bracketSubs(char(M(:)));
        M_flat = M.flatten() if hasattr(M, 'flatten') else M
        M_str = str(M_flat)
        M_char = bracketSubs(M_str)
    else:
        # MATLAB: M_char = char(M(:));
        M_flat = M.flatten() if hasattr(M, 'flatten') else M
        M_char = str(M_flat)
    
    # Convert to Python format
    str_py = _matrix_to_python_code(M_flat)
    
    # MATLAB: str = sprintf('reshape(%s,[%s]);', M_char, num2str(size(M)));
    # Get size as Python tuple
    if isinstance(M, sp.Matrix):
        sz = M.shape
    elif isinstance(M, np.ndarray):
        sz = M.shape
    else:
        sz = M.shape
    
    sz_str = ', '.join(str(s) for s in sz)
    # MATLAB: fprintf(fid, '%s = %s;\n\n', varName, str);
    fid.write(f'    {varName} = np.array({str_py}).reshape(({sz_str}))\n\n')


def _matrix_to_python_code(M: Any) -> str:
    """
    Convert sympy Matrix to Python numpy array code string
    
    Args:
        M: sympy Matrix or numpy array
        
    Returns:
        Python code string representing the matrix
    """
    # Convert sympy Matrix to nested list format
    if isinstance(M, sp.Matrix):
        rows = []
        for i in range(M.rows):
            row = []
            for j in range(M.cols):
                elem = M[i, j]
                # Convert sympy expression to Python code string
                elem_str = _sympy_expr_to_python(elem)
                row.append(elem_str)
            rows.append('[' + ', '.join(row) + ']')
        return 'np.array([' + ',\n        '.join(rows) + '])'
    elif isinstance(M, np.ndarray):
        # Convert numpy array to string representation
        return f'np.array({M.tolist()})'
    else:
        # Try to convert to sympy Matrix
        try:
            M = sp.Matrix(M)
            return _matrix_to_python_code(M)
        except:
            return str(M)


def _sympy_expr_to_python(expr: Any) -> str:
    """
    Convert a sympy expression to Python code string
    
    Args:
        expr: sympy expression
        
    Returns:
        Python code string
    """
    import re
    # Use sympy's code generation
    # Replace common sympy functions with numpy equivalents
    code = str(expr)
    
    # Replace sympy function names with numpy equivalents
    replacements = {
        'sin': 'np.sin',
        'cos': 'np.cos',
        'tan': 'np.tan',
        'exp': 'np.exp',
        'log': 'np.log',
        'sqrt': 'np.sqrt',
        'Abs': 'np.abs',
        'abs': 'np.abs',
    }
    
    for sympy_name, numpy_name in replacements.items():
        # Replace function calls (e.g., sin(x) -> np.sin(x))
        pattern = r'\b' + re.escape(sympy_name) + r'\s*\('
        code = re.sub(pattern, numpy_name + '(', code)
    
    return code


def _convert_matlab_indexing_to_python(code: str) -> str:
    """
    Convert MATLAB-style indexing x(1) to Python indexing x[0, 0]
    Note: In CORA, x and u are always 2D column vectors (shape (n, 1)),
    so x(1) in MATLAB corresponds to x[0, 0] in Python
    
    Args:
        code: Python code string with potential MATLAB indexing
        
    Returns:
        Code string with Python indexing
    """
    import re
    
    # Convert MATLAB-style indexing x(1) to Python indexing x[0, 0]
    # Pattern: variable name followed by (number) - this is MATLAB 1-based indexing
    # In CORA, variables are 2D column vectors, so x(1) -> x[0, 0], x(2) -> x[1, 0], etc.
    # Match patterns like: x(1), xL1R(2), u(1), etc.
    # But avoid matching function calls like sin(x), sqrt(x), np.sqrt(x), etc.
    def matlab_to_python_index(match):
        var_name = match.group(1)  # Variable name (e.g., 'x', 'xL1R', 'u')
        index = int(match.group(2))  # MATLAB 1-based index
        python_index = index - 1  # Convert to Python 0-based index
        # For 2D column vectors, use [index, 0]
        return f'{var_name}[{python_index}, 0]'
    
    # Match variable names (can include L/R for bracket notation) followed by (number)
    # Exclude common function names and numpy functions to avoid converting function calls
    # Pattern: word boundary, variable name (can have L/R), opening paren, number, closing paren
    # Negative lookbehind to exclude function names (np., function names, etc.)
    # Match: identifier followed by (number) but not if preceded by np. or function name
    pattern = r'(?<![a-zA-Z_\.])([a-zA-Z_][a-zA-Z0-9_]*L?\d*R?)\s*\(\s*(\d+)\s*\)'
    code = re.sub(pattern, matlab_to_python_index, code)
    
    return code

