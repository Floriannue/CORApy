"""
evalNthTensor - evaluates the taylor term that corresponds to the
                specified tensor

Syntax:
    T = evalNthTensor(f,x,order)

Inputs:
    T - tensor (symbolic or numeric)
    x - variable values for which the taylor-term is evaluated
    order - order of the tensor T

Outputs:
    res - value of the taylor term

Example: 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: 

Authors:       Niklas Kochdumper
Written:       08-February-2018
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import sympy as sp
import numpy as np
from typing import Any, List
import math


def evalNthTensor(T: List[Any], x: Any, order: int) -> Any:
    """
    Evaluates the taylor term that corresponds to the specified tensor
    
    Args:
        T: tensor (symbolic or numeric, list of tensors)
        x: variable values for which the taylor-term is evaluated (list or array)
        order: order of the tensor T
        
    Returns:
        res: value of the taylor term (list or array)
    """
    
    # Convert x to list/array if needed
    if isinstance(x, (sp.Matrix, np.ndarray)):
        x_list = x.tolist() if hasattr(x, 'tolist') else list(x)
    elif isinstance(x, (list, tuple)):
        x_list = list(x)
    else:
        x_list = [x]
    
    # Convert to numpy array for easier manipulation
    x_arr = np.array(x_list)
    
    # initialize the variable that stores the resulting values of the
    # taylor term
    # MATLAB: res = repmat(x(1),[length(T),1]);
    res = np.array([x_arr[0]] * len(T))

    # different initialization of the algorithm depending on whether the
    # tensor order is odd or even
    # MATLAB: if mod(order,2) == 1
    if order % 2 == 1:  # odd tensor order
        
        # loop over all system dimensions
        # MATLAB: for i = 1:length(T)
        for i in range(len(T)):
            # MATLAB: temp = T{i};
            temp = T[i]
            
            # first-order is a special case, since the derivative there is
            # stored as a matrix instead of a cell array
            # MATLAB: if order == 1
            if order == 1:
                # MATLAB: res(i) = temp * x;
                if isinstance(temp, sp.Matrix):
                    res[i] = (temp * sp.Matrix(x_arr)).tolist()[0][0]
                else:
                    res[i] = np.dot(temp, x_arr)
            else:
                # calculate the value of the term with a recursive function
                # MATLAB: res(i) = x(1) * aux_evalQuadratic(temp{1},x);
                if isinstance(temp[0], (list, tuple)):
                    res[i] = x_arr[0] * aux_evalQuadratic(temp[0], x_arr)
                else:
                    res[i] = x_arr[0] * aux_evalQuadratic(temp[0], x_arr)

                # MATLAB: for j = 2:length(x)
                for j in range(1, len(x_arr)):
                    # MATLAB: res(i) = res(i) + x(j) * aux_evalQuadratic(temp{j},x);
                    if isinstance(temp[j], (list, tuple)):
                        res[i] = res[i] + x_arr[j] * aux_evalQuadratic(temp[j], x_arr)
                    else:
                        res[i] = res[i] + x_arr[j] * aux_evalQuadratic(temp[j], x_arr)
        
    else:  # even tensor order
        
        # loop over all system dimensions
        # MATLAB: for i = 1:length(T)
        for i in range(len(T)):
            # call of the recursive function
            # MATLAB: res(i) = aux_evalQuadratic(T{i},x);
            res[i] = aux_evalQuadratic(T[i], x_arr)
    
    # multiply by factorial factor to obtain the final result
    # MATLAB: res = 1/factorial(order) * res;
    res = (1.0 / math.factorial(order)) * res
    
    return res


# Auxiliary functions -----------------------------------------------------

def aux_evalQuadratic(T: Any, x: np.ndarray) -> Any:
    """
    Recursive function that evaluates the value of the taylor term that
    corresponds to the tensor T at the point x
    
    Args:
        T: tensor (can be matrix or nested list)
        x: variable values (numpy array)
        
    Returns:
        res: evaluated value
    """
    
    # MATLAB: if iscell(T)
    if isinstance(T, (list, tuple)):  # next recursion level
        
        # exploit symmetry in the tensors due to Schwarz's theorem to
        # speed up the computations
        # MATLAB: H = repmat(x(1),[length(x),length(x)]);
        H = np.array([[x[0] for _ in range(len(x))] for _ in range(len(x))])
        # MATLAB: for i = 1:length(x)
        for i in range(len(x)):
            # MATLAB: H(i,i) = aux_evalQuadratic(T{i,i},x);
            H[i, i] = aux_evalQuadratic(T[i][i], x)
            # MATLAB: for j = i+1:length(x)
            for j in range(i + 1, len(x)):
                # MATLAB: temp = aux_evalQuadratic(T{i,j},x);
                temp = aux_evalQuadratic(T[i][j], x)
                # MATLAB: H(i,j) = temp;
                H[i, j] = temp
                # MATLAB: H(j,i) = temp;
                H[j, i] = temp
       
        # MATLAB: res = transpose(x)* H * x;
        res = np.dot(np.dot(x, H), x)
        
    else:  # end of the recursion
        # MATLAB: res = transpose(x)* T * x;
        if isinstance(T, sp.Matrix):
            x_mat = sp.Matrix(x)
            res = (x_mat.T * T * x_mat)[0, 0]
        else:
            res = np.dot(np.dot(x, T), x)
    
    return res

