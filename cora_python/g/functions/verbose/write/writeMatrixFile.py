"""
writeMatrixFile - creates a .py file with a given name on a given path
   that outputs an array of nD matrices; the generated file looks like:
   "   
   def fname(in1, in2, ...):
       out1 = [...]
       out2 = [...]
       ...
       return out1, out2, ...
   "
   where the output variables out1, out2, ... arbitrarily depend on the
   input variables in1, in2, ... 

Syntax:
    writeMatrixFile(M,path,fname)
    writeMatrixFile(M,path,fname,...)
       <'VarNamesIn',varNamesIn> - variable names for input arguments
       <'VarNamesOut',varNamesOut> - variable names for output arguments
       <'BracketSubs',bracketSubsOn> - true/false whether bracketSubs
           should be called
       <'IntervalArithmetic',intervalOn> - true/false whether output
           matrix should be converted to an interval
       <'Sparse',sparseOn> - true/false whether matrix should be sparse

Inputs:
    M - list of matrices
    path - path where the function should be created
    fname - function name for the file computing the Jacobian
    varNamesIn - list with variable names for file input arguments
    varNamesOut - list with variable names for file output arguments
    bracketSubsOn - true/false whether bracketSubs should be called

Outputs:
    handle - function handle to file (callable)

Example:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: derive

Authors:       Mark Wetzlinger
Written:       13-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import importlib.util
from typing import List, Any, Callable, Optional, Tuple
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.checkNameValuePairs import checkNameValuePairs
from cora_python.g.functions.matlab.validate.preprocessing.readNameValuePair import readNameValuePair
from cora_python.g.functions.verbose.write.writeMatrix import writeMatrix
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def writeMatrixFile(M: List[Any], path: str, fname: str, *varargin) -> Callable:
    """
    Creates a .py file with a given name on a given path that outputs an array of nD matrices
    
    Args:
        M: list of matrices (cell array in MATLAB)
        path: path where the function should be created
        fname: function name for the file computing the matrices
        *varargin: optional name-value pairs:
            'VarNamesIn': list - variable names for input arguments
            'VarNamesOut': list - variable names for output arguments
            'BracketSubs': bool - whether bracketSubs should be called
            'IntervalArithmetic': bool - whether output matrix should be converted to an interval
            'Sparse': bool - whether matrix should be sparse
        
    Returns:
        handle: function handle (callable) to the generated file
    """
    
    # MATLAB: narginchk(3,Inf);
    # MATLAB: if mod(nargin,2) ~= 1
    if len(varargin) % 2 != 0:
        raise CORAerror('CORA:oddNumberInputArgs')
    
    # MATLAB: inputArgsCheck({{M,'att','cell'};...})
    if not isinstance(M, list):
        raise TypeError("M must be a list (cell array)")
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(fname, str):
        raise TypeError("fname must be a string")
    
    # read input arguments
    # MATLAB: NVpairs = varargin(1:end);
    NVpairs = list(varargin)
    # check list of name-value pairs
    # MATLAB: checkNameValuePairs(NVpairs,{'VarNamesIn','VarNamesOut',...})
    checkNameValuePairs(NVpairs, ['VarNamesIn', 'VarNamesOut',
                                  'IntervalArithmetic', 'BracketSubs', 'Sparse'])
    
    # interval arithmetic true/false?
    # MATLAB: [NVpairs,bracketSubsOn] = readNameValuePair(NVpairs,'BracketSubs',@islogical,false);
    NVpairs, bracketSubsOn = readNameValuePair(NVpairs, 'BracketSubs', lambda x: isinstance(x, bool), False)
    
    # interval arithmetic true/false?
    # MATLAB: [NVpairs,intervalOn] = readNameValuePair(NVpairs,'IntervalArithmetic',@islogical,false);
    NVpairs, intervalOn = readNameValuePair(NVpairs, 'IntervalArithmetic', lambda x: isinstance(x, bool), False)
    
    # verbose output?
    # MATLAB: [NVpairs,sparseOn] = readNameValuePair(NVpairs,'Sparse',@islogical,false);
    NVpairs, sparseOn = readNameValuePair(NVpairs, 'Sparse', lambda x: isinstance(x, bool), False)
    
    # names for input/output arguments given?
    # MATLAB: [NVpairs,varNamesIn] = readNameValuePair(NVpairs,'VarNamesIn');
    NVpairs, varNamesIn = readNameValuePair(NVpairs, 'VarNamesIn')
    # MATLAB: [NVpairs,varNamesOut] = readNameValuePair(NVpairs,'VarNamesOut');
    NVpairs, varNamesOut = readNameValuePair(NVpairs, 'VarNamesOut')
    
    # ...their default values are a bit more complicated
    # MATLAB: [varNamesIn,varNamesOut] = aux_setDefaultValues(varNamesIn,varNamesOut);
    varNamesIn, varNamesOut = aux_setDefaultValues(varNamesIn, varNamesOut, M)
    
    # ensure that matrices and variable names are of equal length
    # MATLAB: numVars = numel(M);
    numVars = len(M)
    # MATLAB: if numVars ~= numel(varNamesOut)
    if numVars != len(varNamesOut):
        raise CORAerror('CORA:wrongValue', 'second',
                       'Number of variable names must match number of matrices.')
    
    # open file
    # MATLAB: fid = fopen([path filesep fname '.m'],'w');
    file_path = os.path.join(path, f'{fname}.py')
    fid = open(file_path, 'w')
    
    # try-catch to ensure that file will be closed
    try:
        # Write Python imports
        fid.write('import numpy as np\n')
        if sparseOn:
            fid.write('import scipy.sparse\n')
        if intervalOn:
            fid.write('from cora_python.contSet.interval import Interval\n')
        fid.write('\n')
        
        # write first line
        # MATLAB: fprintf(fid, 'function [%s] = %s(%s)\n\n', ...)
        # Python function signature
        varNamesOut_str = ', '.join(varNamesOut)
        varNamesIn_str = ', '.join(varNamesIn)
        fid.write(f'def {fname}({varNamesIn_str}):\n')
        
        # write matrices to file
        # MATLAB: arrayfun(@(i) writeMatrix(fid,M{i},varNamesOut{i},...), 1:numVars);
        for i in range(numVars):
            writeMatrix(fid, M[i], varNamesOut[i],
                       'BracketSubs', bracketSubsOn,
                       'IntervalArithmetic', intervalOn,
                       'Sparse', sparseOn)
        
        # properly end file
        # MATLAB: fprintf(fid, 'end\n');
        # Python return statement
        if len(varNamesOut) == 1:
            fid.write(f'    return {varNamesOut[0]}\n')
        else:
            fid.write(f'    return {varNamesOut_str}\n')
        
    except Exception as ME:
        # close file
        fid.close()
        # MATLAB: rethrow(ME)
        raise ME
    
    # close file
    fid.close()
    
    # output argument
    # MATLAB: handle = eval(['@', fname]);
    # In Python, we need to import the module and get the function
    # Create a function handle by importing the generated module
    try:
        spec = importlib.util.spec_from_file_location(fname, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        handle = getattr(module, fname)
    except Exception as e:
        raise CORAerror('CORA:specialError', 
                       f'Could not create function handle for {fname}: {e}')
    
    return handle


# Auxiliary functions -----------------------------------------------------

def aux_setDefaultValues(varNamesIn: Optional[List[str]], 
                        varNamesOut: Optional[List[str]], 
                        M: List[Any]) -> Tuple[List[str], List[str]]:
    """
    Set default values for variable names
    
    Args:
        varNamesIn: optional list of input variable names
        varNamesOut: optional list of output variable names
        M: list of matrices
        
    Returns:
        varNamesIn: list of input variable names (with defaults if needed)
        varNamesOut: list of output variable names (with defaults if needed)
    """
    
    # by default, the number of input arguments is 1
    # MATLAB: if isempty(varNamesIn)
    if varNamesIn is None or len(varNamesIn) == 0:
        # MATLAB: varNamesIn = {'in1'};
        varNamesIn = ['in1']
    
    # by default, the number of output arguments is equal to the number of
    # Jacobians
    # MATLAB: if isempty(varNamesIn)  <-- NOTE: This is a bug in MATLAB, should be varNamesOut
    # We fix it here:
    if varNamesOut is None or len(varNamesOut) == 0:
        # MATLAB: varNamesOut = arrayfun(@(i) sprintf('out%i',i),1:numel(J),...)
        # NOTE: MATLAB code references 'J' which doesn't exist - should be 'M'
        varNamesOut = [f'out{i+1}' for i in range(len(M))]
    
    return varNamesIn, varNamesOut

