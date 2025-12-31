"""
writeHessianTensorFile - generates a Python file that allows to compute the
   hessian tensor

Syntax:
    writeHessianTensorFile(J2dyn,J2con,path,fname,vars,infsupFlag,options)

Inputs:
    J2dyn - Hessians of differential equation
    J2con - Hessians of constraint equation
    path - path for saving the file
    fname - function name for the Hessian file
    vars - symbolic variables (dict with keys 'x', 'y', 'u', 'p')
    infsupFlag - true if interval arithmetic, otherwise false
    options - additional information for tensors

Outputs:
    - 

Example:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: derivatives

Authors:       Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Written:       21-August-2012
Last update:   08-March-2017
               05-November-2017
               03-December-2017
               13-March-2020 (NK, implemented options.simplify = optimize)
               01-February-2021 (MW, different filename due to infsupFlag)
               01-June-2022 (MW, optimize for constraint part)
               09-June-2022 (MA, considered case that only constraint part is non-empty)
               10-October-2023 (TL, fix missing end in optimized function)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import numpy as np
import sympy as sp
from typing import Any, Dict, List, TextIO, Tuple, Optional
from cora_python.g.functions.verbose.write.writeSparseMatrix import writeSparseMatrix
from cora_python.g.functions.verbose.write.writeSparseMatrixOptimized import writeSparseMatrixOptimized
from cora_python.g.functions.verbose.write.matlabFunction import matlabFunction


def writeHessianTensorFile(J2dyn: Any, J2con: Any, path: str, fname: str,
                          vars: Dict[str, Any], infsupFlag: bool, options: Dict[str, Any]) -> None:
    """
    Generates a Python file that allows to compute the hessian tensor
    
    Args:
        J2dyn: Hessians of differential equation (3D array or list)
        J2con: Hessians of constraint equation (3D array or list)
        path: path for saving the file
        fname: function name for the Hessian file
        vars: dict with symbolic variables (keys: 'x', 'y', 'u', 'p')
        infsupFlag: True if interval arithmetic, otherwise False
        options: dict with additional information for tensors
    """
    
    # MATLAB: [taylMod,opt] = aux_setOptions(options);
    taylMod, opt = aux_setOptions(options)
    
    # squeeze dynamic and constraint part
    # MATLAB: Hdyn = aux_squeeze(J2dyn);
    Hdyn = aux_squeeze(J2dyn)
    # MATLAB: Hcon = aux_squeeze(J2con);
    Hcon = aux_squeeze(J2con)
    
    # open file
    # MATLAB: fid = fopen([path filesep fname '.m'],'w');
    file_path = os.path.join(path, f'{fname}.py')
    fid = open(file_path, 'w')
    
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
        
        # MATLAB: fprintf(fid,'function %s = %s%s\n\n',argsOut,fname,argsIn);
        # Python function signature
        fid.write(f'def {fname}{argsIn}:\n')
        
        # write call to optimized function to reduce the number of interval operations
        # MATLAB: [outDyn,indDyn] = aux_funOptimize(fid,Hdyn,vars.x,opt,'Dyn',argsIn);
        outDyn, indDyn = aux_funOptimize(fid, Hdyn, vars.get('x'), opt, 'Dyn', argsIn)
        # MATLAB: [outCon,indCon] = aux_funOptimize(fid,Hcon,vars.y,opt,'Con',argsIn);
        outCon, indCon = aux_funOptimize(fid, Hcon, vars.get('y'), opt, 'Con', argsIn)
        
        # MATLAB: aux_writeTensors(fid,Hdyn,indDyn,vars.x,opt,taylMod,infsupFlag,'Hf');
        aux_writeTensors(fid, Hdyn, indDyn, vars.get('x'), opt, taylMod, infsupFlag, 'Hf')
        # MATLAB: aux_writeTensors(fid,Hcon,indCon,vars.y,opt,taylMod,infsupFlag,'Hg');
        aux_writeTensors(fid, Hcon, indCon, vars.get('y'), opt, taylMod, infsupFlag, 'Hg')
        
        # add return statement
        # MATLAB: fprintf(fid,'\nend');
        if argsOut.startswith('[') and argsOut.endswith(']'):
            # Multiple outputs
            outputs = argsOut[1:-1].replace(' ', '').split(',')
            fid.write(f'\n    return {", ".join(outputs)}\n')
        else:
            # Single output
            fid.write(f'\n    return {argsOut}\n')
        
        # create optimized function to reduce the number of interval operations
        # MATLAB: if opt
        if opt:
            # MATLAB: aux_writefunOptimize(fid,path,'funOptimizeDyn.m',outDyn,vars.x,cellSymVars);
            aux_writefunOptimize(fid, path, 'funOptimizeDyn.py', outDyn, vars.get('x'), cellSymVars)
            # MATLAB: aux_writefunOptimize(fid,path,'funOptimizeCon.m',outCon,vars.y,cellSymVars);
            aux_writefunOptimize(fid, path, 'funOptimizeCon.py', outCon, vars.get('y'), cellSymVars)
    
    except Exception as ME:
        # close file
        fid.close()
        # MATLAB: rethrow(ME)
        raise ME
    
    # close file
    fid.close()


# Auxiliary functions -----------------------------------------------------

def aux_setOptions(options: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Init options for tensor generation
    
    Returns:
        taylMod: True if Taylor model method is used
        opt: True if optimization is enabled
    """
    
    # default options
    # MATLAB: taylMod = false;
    taylMod = False
    # MATLAB: opt = false;
    opt = False
    
    # read out options
    # MATLAB: if isfield(options,'lagrangeRem')
    if 'lagrangeRem' in options:
        # MATLAB: taylMod = isfield(options.lagrangeRem,'method') ...
        #        && ~strcmp(options.lagrangeRem.method,'interval');
        taylMod = ('method' in options['lagrangeRem'] and
                   options['lagrangeRem']['method'] != 'interval')
        # MATLAB: opt = isfield(options.lagrangeRem,'simplify') ...
        #        && strcmp(options.lagrangeRem.simplify,'optimize');
        opt = ('simplify' in options['lagrangeRem'] and
               options['lagrangeRem']['simplify'] == 'optimize')
    
    return taylMod, opt


def aux_squeeze(J2: Any) -> List[Any]:
    """
    Extract matrices from tensor
    
    Args:
        J2: 3D tensor (first dimension is the number of matrices)
        
    Returns:
        H: list of 2D matrices
    """
    
    # MATLAB: H = cell(size(J2,1),1);
    if isinstance(J2, np.ndarray):
        num_matrices = J2.shape[0]
        H = []
        # MATLAB: for k=1:size(J2,1)
        for k in range(num_matrices):
            # MATLAB: H{k} = squeeze(J2(k,:,:));
            H.append(np.squeeze(J2[k, :, :]))
    else:
        # Assume it's already a list
        H = J2
    
    return H


def aux_args(vars: Dict[str, Any]) -> Tuple[str, str, List[Any]]:
    """
    Generate output/input arguments and list of used variables
    
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
        # MATLAB: argsOut = '[Hf,Hg]';
        argsOut = 'Hf, Hg'
    else:
        # MATLAB: argsOut = 'Hf';
        argsOut = 'Hf'
    
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


def aux_writefunOptimize(fid: TextIO, path: str, fname: str, outFun: Any,
                         vars: Any, cellSymVars: List[Any]) -> None:
    """
    Convert symbolic expression to Python function
    
    Args:
        fid: file handle for main file
        path: path for saving files
        fname: function name
        outFun: symbolic expression(s) to convert
        vars: symbolic variables
        cellSymVars: list of symbolic variables
    """
    
    # MATLAB: if ~isempty(vars)
    if vars is not None and (not hasattr(vars, '__len__') or len(vars) > 0):
        # MATLAB: pathOpt = fullfile(path,fname);
        pathOpt = os.path.join(path, fname)
        # MATLAB: matlabFunction(outFun,'File',pathOpt,'Vars',cellSymVars);
        matlabFunction(outFun, File=pathOpt, Vars=cellSymVars)
        
        # fix missing 'end' etc.
        # MATLAB: aux_fix_optimizedFuncs(pathOpt,fid);
        aux_fix_optimizedFuncs(pathOpt, fid)
        
        # delete funOptimizeDyn|Con file
        # MATLAB: delete([path filesep fname]);
        try:
            os.remove(pathOpt)
        except:
            pass


def aux_fix_optimizedFuncs(pathOpt: str, fid: TextIO) -> None:
    """
    Fix missing 'end' in optimized functions and paste into main file
    
    Args:
        pathOpt: path to optimized function file
        fid: file handle for main file
    """
    
    # print text from pathOpt to hessian tensor file
    # MATLAB: text = fileread(pathOpt);
    with open(pathOpt, 'r') as f:
        text = f.read()
    
    # find last non-empty line
    # MATLAB: idx = 0;
    idx = 0
    # MATLAB: while strcmp(text(end-idx),compose('\n')) || strcmp(text(end-idx),compose('\r'))
    while idx < len(text) and text[-(idx+1)] in ['\n', '\r']:
        idx += 1
    
    # check if end/return is missing
    # MATLAB: if ~strcmp(text(end-idx-2:end-idx),'end')
    # In Python, we check for 'return' statement
    text_trimmed = text.rstrip()
    if not text_trimmed.endswith('return') and 'return' not in text_trimmed.split('\n')[-1]:
        # print with missing return
        # MATLAB: fprintf(fid,'\n\n%s\nend\n',text);
        fid.write(f'\n\n{text}\n')
    else:
        # print without extra end
        # MATLAB: fprintf(fid,'\n\n%s\n',text);
        fid.write(f'\n\n{text}\n')


def aux_funOptimize(fid: TextIO, H: List[Any], vars: Any, opt: bool,
                    suffix: str, argsIn: str) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Optimize function evaluation
    
    Args:
        fid: file handle
        H: list of Hessian matrices
        vars: symbolic variables
        opt: True if optimization is enabled
        suffix: suffix for output variable name ('Dyn' or 'Con')
        argsIn: input arguments string
        
    Returns:
        outFun: list of symbolic expressions
        ind: list of dicts with indices
    """
    
    # ensure object consistency
    # MATLAB: outFun = [];
    outFun = []
    # MATLAB: ind = repmat({struct('row',[],'col',[],'index',[])},size(H));
    ind = [{'row': [], 'col': [], 'index': []} for _ in range(len(H))]
    
    # MATLAB: if ~opt || isempty(vars)
    if not opt or vars is None or (hasattr(vars, '__len__') and len(vars) == 0):
        return outFun, ind
    
    # dynamic part
    # MATLAB: fprintf(fid,'out%s = funOptimize%s%s;\n',suffix,suffix,argsIn);
    fid.write(f'    out{suffix} = funOptimize{suffix}{argsIn}\n')
    
    # store indices of nonempty entries
    # MATLAB: counter = 1;
    counter = 1
    # MATLAB: for i = 1:length(H)
    for i in range(len(H)):
        # MATLAB: [r,c] = find(H{i});
        # Find non-zero entries
        if isinstance(H[i], sp.Matrix):
            M_np = np.array(H[i].tolist())
            row_indices, col_indices = np.nonzero(M_np != 0)
            r = (row_indices + 1).tolist()  # MATLAB uses 1-based
            c = (col_indices + 1).tolist()
        else:
            row_indices, col_indices = np.nonzero(H[i] != 0)
            r = (row_indices + 1).tolist()
            c = (col_indices + 1).tolist()
        
        # MATLAB: if ~isempty(r)
        if len(r) > 0:
            # MATLAB: ind{i}.row = r;
            ind[i]['row'] = r
            # MATLAB: ind{i}.col = c;
            ind[i]['col'] = c
            # MATLAB: ind{i}.index = counter:counter + length(r)-1;
            ind[i]['index'] = list(range(counter, counter + len(r)))
            # MATLAB: counter = counter + length(r);
            counter += len(r)
            # MATLAB: for j = 1:length(r)
            for j in range(len(r)):
                # MATLAB: outFun = [outFun;H{i}(r(j),c(j))];
                if isinstance(H[i], sp.Matrix):
                    outFun.append(H[i][r[j]-1, c[j]-1])  # Convert to 0-based
                else:
                    outFun.append(H[i][r[j]-1, c[j]-1])
    
    return outFun, ind


def aux_writeTensors(fid: TextIO, H: List[Any], ind: List[Dict[str, Any]],
                    vars: Any, opt: bool, taylMod: bool, infsupFlag: bool,
                    varName: str) -> None:
    """
    Write tensor matrices to file
    
    Args:
        fid: file handle
        H: list of Hessian matrices
        ind: list of dicts with indices
        vars: symbolic variables
        opt: True if optimization is enabled
        taylMod: True if Taylor model method is used
        infsupFlag: True if interval arithmetic
        varName: variable name ('Hf' or 'Hg')
    """
    
    # dynamic part
    # MATLAB: if isempty(vars)
    if vars is None or (hasattr(vars, '__len__') and len(vars) == 0):
        # MATLAB: fprintf(fid,'\n\n%s = {};\n\n',varName);
        fid.write(f'\n\n    {varName} = []\n\n')
        return
    
    # MATLAB: for k=1:length(H)
    for k in range(len(H)):
        # get matrix size
        # MATLAB: [rows,cols] = size(H{k});
        if isinstance(H[k], sp.Matrix):
            rows, cols = H[k].shape
        else:
            rows, cols = H[k].shape
        
        # MATLAB: sparseStr = sprintf('sparse(%i,%i)',rows,cols);
        sparseStr = f'scipy.sparse.csr_matrix(({rows}, {cols}))'
        # MATLAB: if infsupFlag
        if infsupFlag:
            # MATLAB: str = sprintf('%s{%i} = interval(%s,%s);',varName,k,sparseStr,sparseStr);
            str_line = f'    {varName}[{k}] = Interval({sparseStr}, {sparseStr})\n'
        else:
            # MATLAB: str = sprintf('%s{%i} = %s;',varName,k,sparseStr);
            str_line = f'    {varName}[{k}] = {sparseStr}\n'
        
        # write in file if Hessian is used as Lagrange remainder
        # MATLAB: fprintf(fid, '\n%s\n', str);
        fid.write(f'\n{str_line}')
        
        # write rest of matrix
        # MATLAB: if ~opt
        if not opt:
            # MATLAB: writeSparseMatrix(fid,H{k},...)
            writeSparseMatrix(fid, H[k], f'{varName}[{k}]', infsupFlag and taylMod)
        else:
            # MATLAB: writeSparseMatrixOptimized(fid,ind{k},...)
            writeSparseMatrixOptimized(fid, ind[k], f'{varName}[{k}]', taylMod)

