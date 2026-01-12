"""
inputArgsLength - computes the number of inputs of a function handle

Syntax:
    [count,out] = inputArgsLength(f)
    [count,out] = inputArgsLength(f,inpArgs)

Inputs:
    f - function handle 
    inpArgs - number of input arguments for the function (max. 26)

Outputs:
    count - vector storing the length of each input argument
    out - output dimension of the function handle

Example:
    f = @(x,u) [x(1)*x(5)^2; sin(x(3)) + u(2)];
    inputArgsLength(f)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: nonlinearSys

Authors:       Victor Gassmann, Mark Wetzlinger
Written:       11-September-2020
Last update:   17-June-2022 (MW, speed up when inpArgs exceeds nargin to f)
               08-October-2024 (MW, fix empty function handle case)
Last revision: 20-November-2022 (MW, restucture alternative method)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import sympy as sp
import inspect
from typing import Tuple, Optional, List
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.auxiliary.combinator import combinator


def inputArgsLength(f: callable, *varargin) -> Tuple[List[int], Tuple[int, ...]]:
    """
    Computes the number of inputs of a function handle
    
    Args:
        f: function handle
        *varargin: optional number of input arguments for the function
        
    Returns:
        count: list storing the length of each input argument
        out: output dimension of the function handle (tuple)
    """
    
    # parse input arguments: number of input arguments to function handle
    # MATLAB: narginf = setDefaultValues({nargin(f)},varargin);
    try:
        sig = inspect.signature(f)
        nargin_f = len(sig.parameters)
    except (TypeError, ValueError):
        # Fallback - try to call with empty args to see what happens
        nargin_f = 2  # Default assumption
    
    nargin_f = setDefaultValues([nargin_f], list(varargin))[0]
    
    # try fast way to determine number of inputs (does not work if
    # statements like "length(x)" occur in the dynamic function
    try:
        # create symbolic variables (up to 100 per input argument supported)
        # MATLAB: maxNumVars = 100;
        maxNumVars = 100
        
        # MATLAB: narginvars = sym('x',[maxNumVars,narginf]);
        # Create symbolic variables for each input argument
        narginvars_list = []
        for i in range(nargin_f):
            vars_i = sp.symbols(f'x_{i}_1:{maxNumVars+1}', real=True)
            narginvars_list.append(np.array(vars_i))
        
        # MATLAB: narginvars_cell = num2cell(narginvars,1);
        narginvars_cell = [arr for arr in narginvars_list]
        
        # evaluate function
        # MATLAB: fsym = f(narginvars_cell{1:narginf});
        fsym = f(*narginvars_cell[:nargin_f])
        
        # output dimension of f
        # MATLAB: out = size(fsym);
        if isinstance(fsym, np.ndarray):
            out = fsym.shape
            # MATLAB fix for empty function handle case (08-October-2024)
            # If output is empty array, out should be 0
            if fsym.size == 0:
                # Empty array: MATLAB returns 0 for empty function
                out = (0,)
            elif any(o <= 1 for o in out):
                out = (max(out),)
        elif isinstance(fsym, (list, tuple)):
            out = (len(fsym),) if len(fsym) > 0 else (0,)
        elif hasattr(fsym, 'shape'):
            out = fsym.shape
            if hasattr(fsym, 'size') and fsym.size == 0:
                out = (0,)
            elif any(o <= 1 for o in out):
                out = (max(out),)
        else:
            out = (1,)
        
        # used variables from array of symbolic variables
        # MATLAB: vars = symvar(fsym);
        if isinstance(fsym, np.ndarray):
            vars_used = set()
            for elem in fsym.flatten():
                if hasattr(elem, 'free_symbols'):
                    vars_used.update(elem.free_symbols)
        elif hasattr(fsym, 'free_symbols'):
            vars_used = fsym.free_symbols
        else:
            vars_used = set()
        
        # logical indices for used variables
        # MATLAB: mask = ismember(narginvars,vars);
        count = []
        for i in range(nargin_f):
            max_idx = 0
            for var in vars_used:
                # Check if variable belongs to i-th input
                var_str = str(var)
                if f'x_{i}_' in var_str:
                    # Extract index from variable name (e.g., 'x_0_5' -> 5)
                    try:
                        idx = int(var_str.split('_')[-1])
                        if idx > max_idx:
                            max_idx = idx
                    except (ValueError, IndexError):
                        pass
            count.append(max_idx)
        
        # sanity check: call function with computed number of input arguments
        try:
            inputs = []
            for i in range(nargin_f):
                # If count[i] is 0, use at least 1 element (for functions that require non-empty inputs)
                # MATLAB handles this by ensuring at least 1 element
                num_elements = max(count[i], 1) if count[i] == 0 else count[i]
                inputs.append(narginvars_cell[i][:num_elements])
            f(*inputs)
            # return only now...
            # But adjust count: if we used more than count[i], keep count[i] as is
            # (the function works with more elements, but the minimum required is count[i])
            return count, out
        except:
            pass  # Fall through to alternative method
    except:
        pass  # Fall through to alternative method
    
    # upper bound
    # MATLAB: bound = 1000;
    bound = 1000
    
    # special case: only one input argument
    # MATLAB: if narginf == 1
    if nargin_f == 1:
        # increment number of input arguments until correct
        # MATLAB: count = 0;
        count = 0
        
        # MATLAB: narginvars = sym('x',[bound,narginf]);
        narginvars = sp.symbols(f'x_0_1:{bound+1}', real=True)
        narginvars = np.array(narginvars)
        
        # MATLAB: while count < bound
        while count < bound:
            count += 1
            try:
                # MATLAB: output = f(narginvars(1:count));
                output = f(narginvars[:count])
                
                # get length of symbolic output -> only exit if the output is a
                # vector (required for matrix-vector multiplication like A*x,
                # where A has a fixed size and x should be chosen such that the
                # result is a vector (would not occur if x is scalar!))
                # MATLAB: out = size(output);
                if isinstance(output, (list, tuple)):
                    out = (len(output),)
                elif isinstance(output, np.ndarray):
                    out = output.shape
                elif hasattr(output, 'shape'):
                    out = output.shape
                else:
                    out = (1,)
                
                # MATLAB: if any(out == 1)
                if any(o == 1 for o in out):
                    out = (max(out),)
                    return [count], out
                # MATLAB: else
                # MATLAB:     continue
                # MATLAB: end
            except:
                continue
    
    # MATLAB: maxVal = 3;
    maxVal = 3
    
    # MATLAB: narginvars = sym('x',[bound,narginf]);
    narginvars_list = []
    for i in range(nargin_f):
        vars_i = sp.symbols(f'x_{i}_1:{bound+1}', real=True)
        narginvars_list.append(np.array(vars_i))
    
    # MATLAB: narginvars_cell = num2cell(narginvars,1);
    narginvars_cell = [arr for arr in narginvars_list]
    
    # checked combinations
    # MATLAB: checked_comb = zeros(1,narginf);
    checked_comb = []
    
    # MATLAB: while true
    while True:
        # MATLAB: comb = combinator(maxVal,narginf) - 1;
        # With 2 args, MATLAB combinator defaults to 'p','r' (permutations with repetition)
        comb = combinator(maxVal, nargin_f) - 1
        
        # MATLAB: if ~isempty(checked_comb)
        if len(checked_comb) > 0:
            # logical index which combinations remain
            # MATLAB: comb_logIdx = true(size(comb,1),1);
            comb_logIdx = np.ones(comb.shape[0], dtype=bool)
            
            # find which already checked combinations are in current list
            # MATLAB: for i=1:size(comb,1)
            for i in range(comb.shape[0]):
                # MATLAB: for j=1:size(checked_comb,1)
                for j in range(len(checked_comb)):
                    # MATLAB: if any(all(checked_comb(j,:) == comb(i,:),2))
                    if np.all(checked_comb[j] == comb[i]):
                        comb_logIdx[i] = False
                        break
            
            # remove already checked combinations
            comb = comb[comb_logIdx]
        
        # MATLAB: for i = 1:size(comb,1)
        for i in range(comb.shape[0]):
            # current combination of input arguments
            # MATLAB: curr_comb = comb(i,:);
            curr_comb = comb[i]
            
            # pass symbolic inputs to function handle
            try:
                # MATLAB: input = cell(narginf,1);
                input_list = []
                # MATLAB: for j = 1:narginf
                for j in range(nargin_f):
                    # MATLAB: input{j} = narginvars_cell{j}(1:curr_comb(j));
                    input_list.append(narginvars_cell[j][:curr_comb[j]])
                
                # MATLAB: output = f(input{:});
                output = f(*input_list)
                
                # function evaluation successful!
                
                # get required length for each input argument
                # MATLAB: count = curr_comb;
                count = curr_comb.tolist()
                
                # get length of symbolic output
                # MATLAB: out = size(output);
                if isinstance(output, (list, tuple)):
                    out = (len(output),)
                elif isinstance(output, np.ndarray):
                    out = output.shape
                elif hasattr(output, 'shape'):
                    out = output.shape
                else:
                    out = (1,)
                
                # MATLAB: if any(out == 1)
                if any(o == 1 for o in out):
                    out = (max(out),)
                    return count, out
                # MATLAB: else
                # MATLAB:     continue
            except:
                # proceed to next number of inputs combination
                continue
        
        # append to list of checked combinations
        # MATLAB: checked_comb = [checked_comb; comb];
        checked_comb.extend(comb.tolist())
        
        # limit number of checks
        # MATLAB: if size(checked_comb,1) > bound
        if len(checked_comb) > bound:
            from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
            raise CORAerror('CORA:specialError',
                           'Could not determine length of input arguments!')
        
        # increment number of inputs
        # MATLAB: maxVal = maxVal + 3;
        maxVal += 3


# Auxiliary functions -----------------------------------------------------

def aux_findIdx(vec: np.ndarray) -> int:
    """
    Find index of last true value in a vector, return 0 if not found
    
    Args:
        vec: boolean vector
        
    Returns:
        idx: index of last True value, or 0 if not found
    """
    
    # MATLAB: idx = find(vec,1,'last');
    idx = np.where(vec)[0]
    if len(idx) == 0:
        return 0
    return int(idx[-1]) + 1  # MATLAB uses 1-based indexing

