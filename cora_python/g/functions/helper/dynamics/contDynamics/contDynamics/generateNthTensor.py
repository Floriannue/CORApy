"""
generateNthTensor - generates the N-th order tensor for function f

Syntax:
    T = generateNthTensor(f,vars,order)
    T = generateNthTensor(f,vars,order,Tprev)

Inputs:
    f - symbolic function (sympy expression or list)
    vars - symbolic variables of the function
    order - order of the tensor that is generated
    Tprev - tensor for order-1 (faster computation if specified)

Outputs:
    T - resulting symbolic tensor

Example:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: writeHigherOrderTensorFiles, evalNthTensor

Authors:       Niklas Kochdumper
Written:       08-February-2018
Last update:   02-February-2021 (MW, different handling of varargin)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import sympy as sp
import numpy as np
from typing import Any, List, Optional
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def generateNthTensor(f: Any, vars: Any, order: int, *varargin) -> List[Any]:
    """
    Generates the N-th order tensor for function f
    
    Args:
        f: symbolic function (sympy expression or list of expressions)
        vars: symbolic variables of the function (sympy symbols or list)
        order: order of the tensor that is generated
        *varargin: optional Tprev - tensor for order-1 (faster computation if specified)
        
    Returns:
        T: resulting symbolic tensor (list of tensors, one for each component of f)
    """
    
    # MATLAB: narginchk(3,4);
    # MATLAB: if nargin == 3
    if len(varargin) == 0:
        # MATLAB: Tprev = []; % previous tensor not provided
        Tprev = None
    # MATLAB: elseif nargin == 4
    elif len(varargin) == 1:
        # MATLAB: Tprev = varargin{1};
        Tprev = varargin[0]
        # catch user inputs that are not supported
        # MATLAB: if order == 1
        if order == 1:
            # MATLAB: throw(CORAerror('CORA:wrongValue','third',...))
            raise CORAerror('CORA:wrongValue', 'third',
                           'The computation from previous tensor not supported for order = 1.')
    else:
        raise CORAerror('CORA:wrongValue', 'nargin',
                       f'generateNthTensor expects 3 or 4 arguments, got {3 + len(varargin)}')
    
    # Convert f to list if single expression
    if not isinstance(f, (list, tuple, np.ndarray)):
        f = [f]
    else:
        f = list(f)
    
    # Convert vars to list if single symbol
    if not isinstance(vars, (list, tuple, np.ndarray)):
        vars = [vars]
    else:
        vars = list(vars)
    
    # initialize tensor 
    # MATLAB: T = cell(length(f),1);
    T = [None] * len(f)
    
    # different algorithms depending on whether or not the previous tensor
    # is provided by the user
    # MATLAB: if isempty(Tprev)
    if Tprev is None:  # previous tensor not provided
       
        # different initialization depending on whether the tensor order is
        # odd or even 
        # MATLAB: if mod(order,2) == 1
        if order % 2 == 1:  # odd tensor order

            # loop over all system dimensions
            # MATLAB: for i = 1:length(f)
            for i in range(len(f)):

                # MATLAB: first = jacobian(f(i),vars);
                first = sp.Matrix([f[i]]).jacobian(sp.Matrix(vars))

                # first order tensor is a special case since the derivative
                # is not stored in a cell array
                # MATLAB: if order == 1
                if order == 1:
                    # MATLAB: T{i} = first;
                    T[i] = first
                else:
                    # MATLAB: T{i} = cell(length(first),1);
                    T[i] = [None] * len(first)
                    # MATLAB: for j = 1:length(first)
                    for j in range(len(first)):
                        # call of the recursive function
                        # MATLAB: T{i}{j} = aux_hessianRecursive(first(j),vars,order-1);
                        T[i][j] = aux_hessianRecursive(first[j], vars, order - 1)

        else:  # even tensor order

            # loop over all system dimensions
            # MATLAB: for i = 1:length(f)
            for i in range(len(f)):
                # call of the recursive function
                # MATLAB: T{i} = aux_hessianRecursive(f(i),vars,order);
                T[i] = aux_hessianRecursive(f[i], vars, order)
    
    else:  # previous tensor provided
    
        # use tensor for order-1 to calculate the current tensor
        # MATLAB: Tprev = varargin{1};
        # Already assigned above
        
        # different initialization depending on whether the tensor order is
        # odd or even
        # MATLAB: if mod(order,2) == 1
        if order % 2 == 1:  # odd tensor order

            # loop over all sysem dimensions
            # MATLAB: for i = 1:length(f)
            for i in range(len(f)):

                # MATLAB: T{i} = cell(length(vars),1);
                T[i] = [None] * len(vars)
               
                # MATLAB: for j = 1:length(vars)
                for j in range(len(vars)):
                    # call of the recursive function
                    # MATLAB: T{i}{j} = aux_hessianFromPrevious(Tprev{i},vars(j));
                    T[i][j] = aux_hessianFromPrevious(Tprev[i], vars[j])

        else:  # even tensor order
            
            # loop over all system dimensions
            # MATLAB: for i = 1:length(f)
            for i in range(len(f)):
               
                # second-order tensor is a special case, since it is derived 
                # from the first-order tensor, which is not stored as a cell
                # array
                # MATLAB: if order == 2
                if order == 2:
                    # fill the quadratic matrix with the derivatives of the
                    # first-order tensor
                    # MATLAB: T{i} = repmat(Tprev{1}(1),[length(vars),length(vars)]);
                    # Create matrix filled with first element
                    first_elem = Tprev[0][0, 0] if isinstance(Tprev[0], sp.Matrix) else Tprev[0]
                    T[i] = sp.Matrix([[first_elem for _ in range(len(vars))] for _ in range(len(vars))])
                    
                    # MATLAB: for k = 1:length(vars)
                    for k in range(len(vars)):
                        # MATLAB: T{i}(k,k) = diff(Tprev{i}(k),vars(k));
                        T[i][k, k] = sp.diff(Tprev[i][k], vars[k])
                        # MATLAB: for j = k+1:length(vars)
                        for j in range(k + 1, len(vars)):
                            # MATLAB: temp = diff(Tprev{i}(k),vars(j));
                            temp = sp.diff(Tprev[i][k], vars[j])
                            # MATLAB: T{i}(k,j) = temp;
                            T[i][k, j] = temp
                            # MATLAB: T{i}(j,k) = temp;
                            T[i][j, k] = temp
                else:
                    # fill the quadratic matrix with derivatives computed
                    # from the previous tensor by the call to the recursive
                    # function
                    # MATLAB: T{i} = cell(length(vars));
                    T[i] = [[None for _ in range(len(vars))] for _ in range(len(vars))]
                    # MATLAB: for k = 1:length(vars)
                    for k in range(len(vars)):
                        # MATLAB: T{i}{k,k} = aux_hessianFromPrevious(Tprev{i}{k},vars(k));
                        T[i][k][k] = aux_hessianFromPrevious(Tprev[i][k], vars[k])
                        # MATLAB: for j = k+1:length(vars)
                        for j in range(k + 1, len(vars)):
                            # MATLAB: temp = aux_hessianFromPrevious(Tprev{i}{k},vars(j));
                            temp = aux_hessianFromPrevious(Tprev[i][k], vars[j])
                            # MATLAB: T{i}{k,j} = temp;
                            T[i][k][j] = temp
                            # MATLAB: T{i}{j,k} = temp;
                            T[i][j][k] = temp
    
    return T


# Auxiliary functions -----------------------------------------------------

def aux_hessianRecursive(f: Any, vars: List[Any], order: int) -> Any:
    """
    Recursive function that calculates the tensor of the specified order for function f
    
    Args:
        f: symbolic function
        vars: list of symbolic variables
        order: order of the tensor
        
    Returns:
        H: resulting tensor
    """
    
    # MATLAB: d = hessian(f,vars);
    # Compute Hessian matrix
    d = sp.hessian(f, vars)

    # end of recursion
    # MATLAB: if order == 2
    if order == 2:
        # MATLAB: H = d;
        return d

    # next level of the recursion
    # MATLAB: H = cell(length(vars));
    H = [[None for _ in range(len(vars))] for _ in range(len(vars))]
    # MATLAB: for i = 1:length(vars)
    for i in range(len(vars)):
        # exploit symmetry in the tensors due to Schwarz's theorem to
        # speed up the computations
        # MATLAB: H{i,i} = aux_hessianRecursive(d(i,i),vars,order-2);
        H[i][i] = aux_hessianRecursive(d[i, i], vars, order - 2)
        # MATLAB: for j = i+1:length(vars)
        for j in range(i + 1, len(vars)):
            # MATLAB: H{i,j} = aux_hessianRecursive(d(i,j),vars,order-2);
            H[i][j] = aux_hessianRecursive(d[i, j], vars, order - 2)
            # MATLAB: H{j,i} = H{i,j};
            H[j][i] = H[i][j]
    
    return H


def aux_hessianFromPrevious(fprev: Any, var: Any) -> Any:
    """
    Recursive function the derivative of the tensor "fprev" with respect to
    the variable var
    
    Args:
        fprev: previous tensor (can be matrix or nested list)
        var: symbolic variable to differentiate with respect to
        
    Returns:
        H: derivative tensor
    """
    
    # end of recursion
    # MATLAB: if ~iscell(fprev)
    if not isinstance(fprev, (list, tuple)):
        # MATLAB: H = diff(fprev,var);
        return sp.diff(fprev, var)

    # next level of the recursion
    # MATLAB: H = cell(size(fprev));
    H = [[None for _ in range(len(fprev[0]))] for _ in range(len(fprev))]
    # MATLAB: for i=1:size(H,1)
    for i in range(len(H)):
        # exploit symmetry in the tensors due to Schwarz's theorem to
        # speed up the computations
        # MATLAB: H{i,i} = aux_hessianFromPrevious(fprev{i,i},var);
        H[i][i] = aux_hessianFromPrevious(fprev[i][i], var)
        # MATLAB: for j=i+1:size(H,1)
        for j in range(i + 1, len(H)):
            # MATLAB: H{i,j} = aux_hessianFromPrevious(fprev{i,j},var);
            H[i][j] = aux_hessianFromPrevious(fprev[i][j], var)
            # MATLAB: H{j,i} = H{i,j};
            H[j][i] = H[i][j]
    
    return H

