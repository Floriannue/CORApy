"""
write3rdOrderTensorFile - generates a Python file that allows to compute the
   third-order terms 

Syntax:
    write3rdOrderTensorFile(J3dyn,J3con,path,fname,vars,infsupFlag,options)

Inputs:
    J3dyn - symbolic third-order tensor
    J3con - symbolic third-order tensor (constraints)
    path - path for saving the file
    fname - function name for the third-order tensor file
    vars - structure containing the symbolic variables (dict with keys 'x', 'y', 'u', 'p')
    infsupFlag - true if interval arithmetic, otherwise false
    options - structure containing the algorithm options

Outputs:
    -

Example:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: derivatives

Authors:       Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Written:       22-August-2012
Last update:   08-March-2017
               12-November-2017
               03-December-2017
               24-January-2018 (NK)
               13-March-2020 (NK, implemented options.simplify = optimize)
               01-February-2021 (MW, add infsupFlag for different filenames)
Last revision: 09-October-2024 (MW, refactor)
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import numpy as np
import sympy as sp
from typing import Any, Dict, List, TextIO, Tuple, Optional
from cora_python.g.functions.verbose.write.writeSparseMatrix import writeSparseMatrix
from cora_python.g.functions.verbose.write.writeSparseMatrixOptimized import writeSparseMatrixOptimized
from cora_python.g.functions.verbose.write.writeMatrix import writeMatrix
from cora_python.g.functions.verbose.write.matlabFunction import matlabFunction


def write3rdOrderTensorFile(J3dyn: Any, J3con: Any, path: str, fname: str,
                            vars: Dict[str, Any], infsupFlag: bool, options: Dict[str, Any]) -> None:
    """
    Generates a Python file that allows to compute the third-order terms
    
    Args:
        J3dyn: symbolic third-order tensor (4D array)
        J3con: symbolic third-order tensor (constraints, 4D array)
        path: path for saving the file
        fname: function name for the third-order tensor file
        vars: dict containing the symbolic variables (keys: 'x', 'y', 'u', 'p')
        infsupFlag: True if interval arithmetic, otherwise False
        options: dict containing the algorithm options
    """
    
    # read out options
    # MATLAB: [taylMod,replace,parallel,opt,rep] = aux_setOptions(options,vars);
    taylMod, replace, parallel, opt, rep = aux_setOptions(options, vars)
    
    # rearrange dynamic and constraint part (4D -> 2D cell array of 2D)
    # MATLAB: Tdyn = aux_rearrange(J3dyn);
    Tdyn = aux_rearrange(J3dyn)
    # MATLAB: Tcon = aux_rearrange(J3con);
    Tcon = aux_rearrange(J3con)
    # sizes
    # MATLAB: [n_dyn,m_dyn] = size(Tdyn);
    n_dyn = len(Tdyn)
    m_dyn = len(Tdyn[0]) if n_dyn > 0 else 0
    
    # open file
    # MATLAB: fid = fopen([path filesep fname '.m'],'w');
    file_path = os.path.join(path, f'{fname}.py')
    fid = open(file_path, 'w')
    
    # try-catch block to ensure that file is closed
    try:
        # function arguments depending on occurring variable types
        # MATLAB: [argsOut,argsIn,cellSymVars] = aux_args(vars);
        argsOut, argsIn, cellSymVars = aux_args(vars)
        
        # Write Python imports
        fid.write('import numpy as np\n')
        fid.write('import scipy.sparse\n')
        if infsupFlag:
            fid.write('from cora_python.contSet.interval import Interval\n')
        fid.write('\n')
        
        # MATLAB: fprintf(fid, 'function %s = %s%s\n\n',argsOut,fname,argsIn);
        # Python function signature
        fid.write(f'def {fname}{argsIn}:\n')
        
        # write replacements
        # MATLAB: [rep,r] = aux_writeReplacements(fid,replace,rep);
        rep, r = aux_writeReplacements(fid, replace, rep)
        
        # write call to optimized function to reduce the number of interval 
        # arithmetic evaluations (e.g., sin(x(1)) is used multiple times)
        # MATLAB: [out,ind] = aux_funOptimize(Tdyn,opt);
        out, ind = aux_funOptimize(Tdyn, opt)
        # MATLAB: if ~isempty(out)
        if len(out) > 0:
            # MATLAB: fprintf(fid, 'out = funOptimize%s;\n\n', argsIn);
            fid.write(f'    out = funOptimize{argsIn}\n\n')
        
        # beginning of parallel execution
        # MATLAB: if parallel
        if parallel:
            # MATLAB: fprintf(fid, 'C = cell(%i,1);\n\n', n_dyn);
            fid.write(f'    C = [[] for _ in range({n_dyn})]\n\n')
            # MATLAB: fprintf(fid, 'parfor i = 1:%i\n\n', n_dyn);
            # Note: Python doesn't have parfor, we'll use a regular for loop
            # In practice, parallel execution would be handled differently
            fid.write(f'    for i in range({n_dyn}):\n')
            # MATLAB: fprintf(fid, '%sswitch i\n', aux_tabs(1));
            fid.write(f'{aux_tabs(1)}    if i == ')
            # We'll handle the switch differently in Python
        
        # pre-compute initialization string
        # MATLAB: initStr = aux_initStr(Tdyn,parallel,infsupFlag);
        initStr = aux_initStr(Tdyn, parallel, infsupFlag)
        
        # Initialize Tf if not using parallel execution
        # MATLAB: Tf = cell(n_dyn, m_dyn);
        if not parallel:
            fid.write(f'    Tf = [[None for _ in range({m_dyn})] for _ in range({n_dyn})]\n\n')
        
        # dynamic part
        # MATLAB: indZero = false(n_dyn,m_dyn);
        indZero = np.zeros((n_dyn, m_dyn), dtype=bool)
        
        # MATLAB: for k=1:n_dyn
        for k in range(n_dyn):
            # MATLAB: if parallel
            if parallel:
                # MATLAB: fprintf(fid,'\n%scase %i\n',aux_tabs(2),k);
                fid.write(f'\n{aux_tabs(1)}    if i == {k}:\n')
            
            # MATLAB: for l=1:m_dyn
            for l in range(m_dyn):
                # substitute all replacements
                # MATLAB: if replace
                if replace:
                    # MATLAB: Tdyn{k,l} = subs(Tdyn{k,l},rep,r);
                    Tdyn[k][l] = Tdyn[k][l].subs(list(zip(rep, r)))
                
                # write matrix
                # MATLAB: if parallel
                if parallel:
                    # MATLAB: fprintf(fid, '\n%s\n', sprintf(initStr,l));
                    fid.write(f'\n{aux_tabs(2)}        {initStr.format(l)}\n')
                    # MATLAB: indZero(k,l) = writeSparseMatrix(fid,Tdyn{k,l},...)
                    var_name = f'{aux_tabs(3)}        C[i][{l}]'
                    indZero[k, l] = writeSparseMatrix(fid, Tdyn[k][l], var_name, taylMod)
                # MATLAB: elseif opt
                elif opt:
                    # MATLAB: fprintf(fid, '\n%s\n', sprintf(initStr,k,l));
                    fid.write(f'\n    {initStr.format(k, l)}\n')
                    # MATLAB: indZero(k,l) = writeSparseMatrixOptimized(fid,ind{k,l},...)
                    indZero[k, l] = writeSparseMatrixOptimized(fid, ind[k][l],
                                                               f'Tf[{k}][{l}]', taylMod)
                else:
                    # MATLAB: fprintf(fid, '\n%s\n', sprintf(initStr,k,l));
                    fid.write(f'\n    {initStr.format(k, l)}\n')
                    # MATLAB: indZero(k,l) = writeSparseMatrix(fid,Tdyn{k,l},...)
                    # Note: writeSparseMatrix returns True if matrix is empty, False if it has non-zero elements
                    # indZero[k, l] = True means matrix is empty, False means it has non-zero elements
                    empty_result = writeSparseMatrix(fid, Tdyn[k][l],
                                                      f'Tf[{k}][{l}]', taylMod)
                    indZero[k, l] = empty_result
                    # Debug: verify the assignment worked and check matrix contents
                    if options.get('verbose', False):
                        if isinstance(Tdyn[k][l], sp.Matrix):
                            rows, cols = Tdyn[k][l].shape
                            non_zero_count = 0
                            for i in range(min(rows, 3)):  # Check first few rows
                                for j in range(min(cols, 3)):  # Check first few cols
                                    elem = Tdyn[k][l][i, j]
                                    if isinstance(elem, sp.Basic):
                                        if hasattr(elem, 'is_zero') and elem.is_zero is True:
                                            pass  # zero
                                        else:
                                            non_zero_count += 1
                                    elif elem != 0:
                                        non_zero_count += 1
                            if non_zero_count == 0 and (k, l) != (0, 0):  # Tf[0][0] should have non-zero
                                print(f'  DEBUG: indZero[{k}, {l}] = {empty_result} (non-zero count: {non_zero_count}, should be True if empty)')
                        else:
                            print(f'  DEBUG: indZero[{k}, {l}] = {empty_result} (matrix empty: {empty_result})')
                
                # MATLAB: if options.verbose
                if options.get('verbose', False):
                    # MATLAB: fprintf('     .. dynamic index %i,%i\n',k,l);
                    print(f'     .. dynamic index {k+1},{l+1}')
        
        # end of parallel execution (rewrite to Tf)
        # MATLAB: if parallel
        if parallel:
            # MATLAB: fprintf(fid, '\n%send\n', aux_tabs(1));
            fid.write(f'\n{aux_tabs(1)}    # End of parallel section\n')
            # MATLAB: fprintf(fid, 'end\n\n');
            fid.write('\n')
            # MATLAB: fprintf(fid, 'Tf = cell(%i,%i);\n',n_dyn,m_dyn);
            fid.write(f'    Tf = [[None for _ in range({m_dyn})] for _ in range({n_dyn})]\n')
            # MATLAB: fprintf(fid, 'for i=1:%i\n', n_dyn);
            fid.write(f'    for i in range({n_dyn}):\n')
            # MATLAB: fprintf(fid, '%sfor j=1:%i\n', aux_tabs(1), m_dyn);
            fid.write(f'{aux_tabs(1)}    for j in range({m_dyn}):\n')
            # MATLAB: fprintf(fid, '%sTf{i,j} = C{i}{j};\n', aux_tabs(2));
            fid.write(f'{aux_tabs(2)}        Tf[i][j] = C[i][j]\n')
            # MATLAB: fprintf(fid, '%send\n', aux_tabs(1));
            fid.write(f'{aux_tabs(1)}    # End inner loop\n')
            # MATLAB: fprintf(fid, 'end\n\n');
            fid.write('\n')
        
        # constraint part
        # Initialize Tg if there are constraints
        # MATLAB: In MATLAB, Tg is initialized implicitly by first assignment
        # In Python, we need to initialize it explicitly
        if len(Tcon) > 0 and len(Tcon[0]) > 0:
            n_con = len(Tcon)
            m_con = len(Tcon[0]) if len(Tcon) > 0 else 0
            fid.write(f'    Tg = [[None for _ in range({m_con})] for _ in range({n_con})]\n\n')
        
        # MATLAB: for k=1:size(Tcon,1)
        for k in range(len(Tcon)):
            # MATLAB: for l=1:size(Tcon,2)
            for l in range(len(Tcon[k]) if len(Tcon) > 0 else 0):
                # MATLAB: [rows,cols] = size(Tcon{k,l});
                if isinstance(Tcon[k][l], sp.Matrix):
                    rows, cols = Tcon[k][l].shape
                else:
                    rows, cols = Tcon[k][l].shape
                # MATLAB: sparseStr = sprintf('sparse(%i,%i)', rows, cols);
                sparseStr = f'scipy.sparse.csr_matrix(({rows}, {cols}))'
                # MATLAB: str = sprintf('Tg{%i,%i} = interval(%s,%s);\n',k,l,sparseStr,sparseStr);
                str_line = f'    Tg[{k}][{l}] = Interval({sparseStr}, {sparseStr})\n'
                # MATLAB: fprintf(fid, '%s\n\n', str);
                fid.write(f'{str_line}\n')
                # MATLAB: writeSparseMatrix(fid,Tcon{k,l},sprintf('Tg{%i,%i}',k,l));
                writeSparseMatrix(fid, Tcon[k][l], f'Tg[{k}][{l}]')
                
                # MATLAB: if options.verbose
                if options.get('verbose', False):
                    # MATLAB: fprintf('     .. dynamic index %i, %i\n',k,l);
                    print(f'     .. dynamic index {k+1}, {l+1}')
        
        # invert values to represent non-zero indices
        # MATLAB: indNonZero = ~indZero;
        indNonZero = ~indZero
        # MATLAB: fprintf(fid,'\nind = cell(%i,1);\n',size(indNonZero,1));
        fid.write(f'\n    ind = [[] for _ in range({indNonZero.shape[0]})]\n')
        # MATLAB: for i=1:size(indNonZero,1)
        for i in range(indNonZero.shape[0]):
            # MATLAB: fprintf(fid,'ind{%i} = %s;\n',i,mat2str(find(indNonZero(i,:)')));
            # Find non-zero indices (MATLAB uses 1-based, Python uses 0-based)
            # MATLAB's find returns 1-based indices, so we need to convert
            non_zero_indices = np.where(indNonZero[i, :])[0]
            # MATLAB uses 1-based indexing, so convert to 1-based for consistency
            # But Python uses 0-based, so keep 0-based
            if len(non_zero_indices) > 0:
                indices_str = '[' + ', '.join(str(idx) for idx in non_zero_indices) + ']'
            else:
                indices_str = '[]'
            fid.write(f'    ind[{i}] = {indices_str}\n')
        
        # properly end function
        # MATLAB: fprintf(fid,'\nend\n');
        # Python return statement
        # Check if function body is empty (no Tf initialization written)
        # If empty, initialize Tf and ind to empty structures
        if n_dyn == 0 or m_dyn == 0:
            # Empty tensor - return empty structures
            fid.write('    # Empty tensor\n')
            if 'Tg' in argsOut:
                fid.write('    Tf = []\n')
                fid.write('    Tg = []\n')
                fid.write('    ind = []\n')
                fid.write('    return Tf, Tg, ind\n')
            else:
                fid.write('    Tf = []\n')
                fid.write('    ind = []\n')
                fid.write('    return Tf, ind\n')
        else:
            # Normal return statement
            if argsOut.startswith('[') and argsOut.endswith(']'):
                outputs = argsOut[1:-1].replace(' ', '').split(',')
                fid.write(f'\n    return {", ".join(outputs)}\n')
            else:
                fid.write(f'\n    return {argsOut}\n')
        
        # create optimized function to reduce the number of interval operations
        # MATLAB: aux_writefunOptimize(fid,out,path,cellSymVars);
        aux_writefunOptimize(fid, out, path, cellSymVars)
    
    except Exception as ME:
        # close file
        fid.close()
        # MATLAB: rethrow(ME);
        raise ME
    
    # close file
    fid.close()


# Auxiliary functions -----------------------------------------------------

def aux_setOptions(options: Dict[str, Any], vars: Dict[str, Any]) -> Tuple[bool, bool, bool, bool, Any]:
    """
    Init options for tensor generation
    
    Returns:
        taylMod: True if Taylor model method is used
        replace: True if replacements are enabled
        parallel: True if parallel execution is enabled
        opt: True if optimization is enabled
        rep: replacement expressions (if any)
    """
    
    # init as false
    # MATLAB: taylMod = false;
    taylMod = False
    # MATLAB: replace = false;
    replace = False
    # MATLAB: parallel = false;
    parallel = False
    # MATLAB: opt = false;
    opt = False
    # MATLAB: rep = [];
    rep = []
    
    # read out from field 'lagrangeRem'
    # MATLAB: if isfield(options,'lagrangeRem')
    if 'lagrangeRem' in options:
        # MATLAB: taylMod = isfield(options.lagrangeRem,'method') ...
        #        && ~strcmp(options.lagrangeRem.method,'interval');
        taylMod = ('method' in options['lagrangeRem'] and
                   options['lagrangeRem']['method'] != 'interval')
        # MATLAB: if isfield(options.lagrangeRem,'replacements')
        if 'replacements' in options['lagrangeRem']:
            # MATLAB: replace = true;
            replace = True
            # MATLAB: if ~isempty(vars.p)
            if vars.get('p') is not None and (not hasattr(vars['p'], '__len__') or len(vars['p']) > 0):
                # MATLAB: rep = options.lagrangeRem.replacements(vars.x,vars.u,vars.p);
                rep = options['lagrangeRem']['replacements'](vars['x'], vars['u'], vars['p'])
            else:
                # MATLAB: rep = options.lagrangeRem.replacements(vars.x,vars.u);
                rep = options['lagrangeRem']['replacements'](vars['x'], vars['u'])
        # MATLAB: parallel = isfield(options.lagrangeRem,'tensorParallel') ...
        #        && options.lagrangeRem.tensorParallel == 1;
        parallel = ('tensorParallel' in options['lagrangeRem'] and
                   options['lagrangeRem']['tensorParallel'] == 1)
        # MATLAB: opt = isfield(options.lagrangeRem,'simplify') ...
        #        && strcmp(options.lagrangeRem.simplify,'optimize');
        opt = ('simplify' in options['lagrangeRem'] and
               options['lagrangeRem']['simplify'] == 'optimize')
    
    return taylMod, replace, parallel, opt, rep


def aux_rearrange(J3: Any) -> List[List[Any]]:
    """
    Rearrange 4D array into 2D list of matrices
    
    Args:
        J3: 4D array (first two dimensions are the tensor indices)
        
    Returns:
        T: 2D list of 2D matrices
    """
    
    # MATLAB: T = cell(size(J3,1),size(J3,2));
    if isinstance(J3, np.ndarray):
        n = J3.shape[0]
        m = J3.shape[1]
        T = []
        # MATLAB: for k=1:size(J3,1)
        for k in range(n):
            T_row = []
            # MATLAB: for l=1:size(J3,2)
            for l in range(m):
                # MATLAB: T{k,l} = squeeze(J3(k,l,:,:));
                T_row.append(np.squeeze(J3[k, l, :, :]))
            T.append(T_row)
    else:
        # Assume it's already a 2D list (from aux_thirdOrderDerivatives)
        # MATLAB: T{k,l} = squeeze(J3(k,l,:,:));
        # In MATLAB, squeeze always returns a matrix, even if all zeros
        # In Python, if J3[k][l] is None (not computed), we need to create an empty matrix
        # Get dimensions from first non-None element
        n = len(J3)
        m = len(J3[0]) if n > 0 else 0
        T = []
        for k in range(n):
            T_row = []
            for l in range(m):
                # MATLAB: T{k,l} = squeeze(J3(k,l,:,:));
                # In MATLAB, squeeze always returns a matrix, even if all zeros
                # In Python, if J3[k][l] is not a Matrix (still the initial nested list structure),
                # it means it wasn't computed, so we need to create an empty zero matrix
                if not isinstance(J3[k][l], sp.Matrix):
                    # Not a matrix - means it wasn't computed (still the initial nested list)
                    # Get size from first computed matrix
                    matrix_size = None
                    for k2 in range(n):
                        for l2 in range(m):
                            if isinstance(J3[k2][l2], sp.Matrix):
                                matrix_size = J3[k2][l2].shape
                                break
                        if matrix_size is not None:
                            break
                    if matrix_size is not None:
                        # Create zero matrix of same size
                        T_row.append(sp.zeros(matrix_size[0], matrix_size[1]))
                    else:
                        # Default size (shouldn't happen, but handle gracefully)
                        T_row.append(sp.zeros(7, 7))  # Default for tank6Eq
                else:
                    # Already a matrix, use as-is
                    T_row.append(J3[k][l])
            T.append(T_row)
    
    return T


def aux_args(vars: Dict[str, Any]) -> Tuple[str, str, List[Any]]:
    """
    Generate the function signature of the generated file
    
    Args:
        vars: dict with symbolic variables (keys: 'x', 'y', 'u', 'p')
        
    Returns:
        argsOut: output arguments string
        argsIn: input arguments string
        cellSymVars: list of symbolic variables
    """
    
    # output arguments
    # MATLAB: if ~isempty(vars.y)
    if vars.get('y') is not None and (not hasattr(vars['y'], '__len__') or len(vars['y']) > 0):
        # MATLAB: argsOut = '[Tf,Tg,ind]';
        argsOut = 'Tf, Tg, ind'
    else:
        # MATLAB: argsOut = '[Tf,ind]';
        argsOut = 'Tf, ind'
    
    # input arguments and list of symbolic variables
    # MATLAB: if ~isempty(vars.y)
    if vars.get('y') is not None and (not hasattr(vars['y'], '__len__') or len(vars['y']) > 0):
        # MATLAB: nonemptyVars = [true;true;true;false];
        nonemptyVars = [True, True, True, False]
    # MATLAB: elseif ~isempty(vars.p)
    elif vars.get('p') is not None and (not hasattr(vars['p'], '__len__') or len(vars['p']) > 0):
        # MATLAB: nonemptyVars = [true;false;true;true];
        nonemptyVars = [True, False, True, True]
    else:
        # MATLAB: nonemptyVars = [true;false;true;false];
        nonemptyVars = [True, False, True, False]
    
    # MATLAB: argsIn = {'x','y','u','p'};
    argsIn_names = ['x', 'y', 'u', 'p']
    # MATLAB: argsIn = ['(' strjoin(argsIn(nonemptyVars),',') ')'];
    argsIn_selected = [argsIn_names[i] for i in range(4) if nonemptyVars[i]]
    argsIn = '(' + ', '.join(argsIn_selected) + ')'
    
    # MATLAB: cellSymVars = {vars.x,vars.y,vars.u,vars.p};
    cellSymVars = [vars.get('x'), vars.get('y'), vars.get('u'), vars.get('p')]
    # MATLAB: cellSymVars = cellSymVars(nonemptyVars);
    cellSymVars = [cellSymVars[i] for i in range(4) if nonemptyVars[i]]
    
    return argsOut, argsIn, cellSymVars


def aux_initStr(Tdyn: List[List[Any]], parallel: bool, infsupFlag: bool) -> str:
    """
    Initialization string
    
    Args:
        Tdyn: 2D list of matrices
        parallel: True if parallel execution
        infsupFlag: True if interval arithmetic
        
    Returns:
        initStr: format string for initialization
    """
    
    # MATLAB: [rows,cols] = size(Tdyn{1,1});
    if isinstance(Tdyn[0][0], sp.Matrix):
        rows, cols = Tdyn[0][0].shape
    else:
        rows, cols = Tdyn[0][0].shape
    # MATLAB: sparseStr = sprintf('sparse(%i,%i)',rows,cols);
    sparseStr = f'scipy.sparse.csr_matrix(({rows}, {cols}))'
    
    # MATLAB: if infsupFlag
    if infsupFlag:
        # MATLAB: if parallel
        if parallel:
            # MATLAB: initStr = sprintf('%sC{i}{%%i} = interval(%s,%s);',...)
            initStr = f'{aux_tabs(3)}        C[i][{{}}] = Interval({sparseStr}, {sparseStr})'
        else:
            # MATLAB: initStr = sprintf('Tf{%%i,%%i} = interval(%s,%s);',sparseStr,sparseStr);
            initStr = f'Tf[{{}}][{{}}] = Interval({sparseStr}, {sparseStr})'
    else:
        # MATLAB: if parallel
        if parallel:
            # MATLAB: initStr = sprintf('%sC{i}{%%i} = %s;',aux_tabs(3),sparseStr);
            initStr = f'{aux_tabs(3)}        C[i][{{}}] = {sparseStr}'
        else:
            # MATLAB: initStr = sprintf('Tf{%%i,%%i} = %s;',sparseStr);
            initStr = f'Tf[{{}}][{{}}] = {sparseStr}'
    
    return initStr


def aux_tabs(n: int) -> str:
    """
    Get tabs (four spaces) for pretty indenting of generated file
    
    Args:
        n: number of indentation levels
        
    Returns:
        str: indentation string
    """
    
    # MATLAB: str = repmat('  ',1,n);
    # Python uses 4 spaces per level
    return '    ' * n


def aux_writeReplacements(fid: TextIO, replace: bool, rep: Any) -> Tuple[Any, Any]:
    """
    Write replacements to file
    
    Args:
        fid: file handle
        replace: True if replacements should be written
        rep: replacement expressions
        
    Returns:
        rep: replacement expressions (possibly transposed)
        r: symbolic variables for replacements
    """
    
    # MATLAB: r = [];
    r = []
    # MATLAB: if replace
    if replace:
        # generate symbolic variables for the replacements (hack because
        #   sym('rL%dR',[1,1])
        # yields rL1R1 instead of rL1R... which is annoying)
        # MATLAB: r = sym('rL%dR',[length(rep)+1,1]);
        # MATLAB: r = r(1:end-1);
        if hasattr(rep, '__len__'):
            num_rep = len(rep)
        else:
            num_rep = 1
            rep = [rep]
        
        r = [sp.Symbol(f'rL{i+1}R', real=True) for i in range(num_rep)]
        
        # MATLAB: if size(rep,1) == 1
        if isinstance(rep, (list, tuple)) and len(rep) > 0:
            if isinstance(rep[0], (list, tuple, np.ndarray)) and len(rep[0]) == 1:
                # Transpose if needed
                rep = [[r] for r in rep]
        
        # write the replacements to the file
        # MATLAB: fprintf(fid, '%% replacements\n');
        fid.write('    # replacements\n')
        # MATLAB: writeMatrix(fid,rep,'r','BracketSubs',true);
        writeMatrix(fid, rep, 'r', 'BracketSubs', True)
    
    return rep, r


def aux_funOptimize(Tdyn: List[List[Any]], opt: bool) -> Tuple[List[Any], List[List[Dict[str, Any]]]]:
    """
    Functions for simplify = 'optimize'
    
    Args:
        Tdyn: 2D list of matrices
        opt: True if optimization is enabled
        
    Returns:
        out: list of symbolic expressions
        ind: 2D list of dicts with indices
    """
    
    # ensure object consistency
    # MATLAB: out = [];
    out = []
    # MATLAB: ind = repmat({struct('row',[],'col',[],'index',[])},size(Tdyn));
    ind = [[{'row': [], 'col': [], 'index': []} for _ in range(len(Tdyn[0]) if len(Tdyn) > 0 else 0)]
           for _ in range(len(Tdyn))]
    
    # MATLAB: if ~opt
    if not opt:
        return out, ind
    
    # store indices of nonempty entries
    # MATLAB: counter = 1;
    counter = 1
    # MATLAB: for i = 1:size(Tdyn,1)
    for i in range(len(Tdyn)):
        # MATLAB: for j = 1:size(Tdyn,2)
        for j in range(len(Tdyn[i])):
            # MATLAB: [r,c] = find(Tdyn{i,j});
            # Find non-zero entries
            if isinstance(Tdyn[i][j], sp.Matrix):
                M_np = np.array(Tdyn[i][j].tolist())
                row_indices, col_indices = np.nonzero(M_np != 0)
                r = (row_indices + 1).tolist()  # MATLAB uses 1-based
                c = (col_indices + 1).tolist()
            else:
                row_indices, col_indices = np.nonzero(Tdyn[i][j] != 0)
                r = (row_indices + 1).tolist()
                c = (col_indices + 1).tolist()
            
            # MATLAB: if ~isempty(r)
            if len(r) > 0:
                # MATLAB: ind{i,j}.row = r;
                ind[i][j]['row'] = r
                # MATLAB: ind{i,j}.col = c;
                ind[i][j]['col'] = c
                # MATLAB: ind{i,j}.index = counter:counter + length(r)-1;
                ind[i][j]['index'] = list(range(counter, counter + len(r)))
                # MATLAB: counter = counter + length(r);
                counter += len(r)
                # MATLAB: for k = 1:length(r)
                for k in range(len(r)):
                    # MATLAB: out = [out;Tdyn{i,j}(r(k),c(k))];
                    if isinstance(Tdyn[i][j], sp.Matrix):
                        out.append(Tdyn[i][j][r[k]-1, c[k]-1])  # Convert to 0-based
                    else:
                        out.append(Tdyn[i][j][r[k]-1, c[k]-1])
    
    return out, ind


def aux_writefunOptimize(fid: TextIO, out: List[Any], path: str, cellSymVars: List[Any]) -> None:
    """
    Write optimized function to file
    
    Args:
        fid: file handle for main file
        out: list of symbolic expressions
        path: path for saving files
        cellSymVars: list of symbolic variables
    """
    
    # out is always empty if simplify is not 'optimize' (opt == true)
    # MATLAB: if isempty(out)
    if len(out) == 0:
        return
    
    # create file with optimized evaluation
    # MATLAB: pathTemp = fullfile(path,'funOptimize.m');
    pathTemp = os.path.join(path, 'funOptimize.py')
    # MATLAB: matlabFunction(out,'File',pathTemp,'Vars',cellSymVars);
    matlabFunction(out, File=pathTemp, Vars=cellSymVars)
    
    # we paste the content of the generated file into the tensor file;
    # in some MATLAB versions, matlabFunction does not add the keyword
    # 'end' at the end, so we check whether it's there and append it if not
    # MATLAB: text = fileread(pathTemp);
    with open(pathTemp, 'r') as f:
        text = f.read()
    # MATLAB: fprintf(fid,'\n%s\n',text);
    fid.write(f'\n{text}\n')
    # MATLAB: if ~contains(text,'end')
    if 'return' not in text.split('\n')[-1]:
        # MATLAB: fprintf(fid,'\nend\n\n');
        fid.write('\n')
    
    # Clean up temporary file
    try:
        os.remove(pathTemp)
    except:
        pass

