"""
full_fact - gives full factorial design matrix for any levels  
   more than 2 of any number of variables (minimum 2)

Syntax:
    des_mat = full_fact(x1, x2, x3)
    des_mat = full_fact([-1, 1], [100, 200, 300], [1, 2, 3, 4])

Inputs:
    x1, x2, x3 - variable levels either in row or column vector. These are
                not number of levels but the levels itself

Outputs:
    des_mat - mxn matrix where
      m = total number of designs (product of all levels) and  
      n = number of variables  
           The first column shows all the first variable levels, second 
           column shows second variable levels and so on.

Example:
    x1 = [-1, 1]
    x2 = [100, 200, 300]
    des_mat = full_fact(x1, x2)  # OR
    des_mat = full_fact([-1, 1], [100, 200, 300])

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors: Bhaskar Dongare
         Python translation by AI Assistant
Written: 17-November-2008
Last update: 17-June-2022 (MW, formatting)
Last revision: ---
"""

import numpy as np
from typing import List, Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def full_fact(*args) -> np.ndarray:
    """
    Gives full factorial design matrix for any levels more than 2 of any number of variables.
    
    Args:
        *args: Variable levels either in row or column vector. These are not number 
               of levels but the levels itself
        
    Returns:
        des_mat: mxn matrix where m = total number of designs (product of all levels) 
                and n = number of variables
                
    Raises:
        CORAerror: If less than 2 arguments or if any variable has less than 2 levels
    """
    if len(args) < 2:
        raise ValueError("Minimum 2 variables required")
    
    # Convert inputs to column vectors and get levels
    varargin = []
    levels = []
    
    for i, arg in enumerate(args):
        # Convert to numpy array
        var = np.array(arg)
        
        # Convert row vector to column vector
        if var.ndim == 1:
            var = var.reshape(-1, 1)
        elif var.ndim == 0:
            var = var.reshape(1, 1)
        elif var.shape[0] == 1:
            var = var.T
            
        varargin.append(var)
        levels.append(len(var))
    
    # Check number of variables and levels of each variable
    if not all(level > 1 for level in levels):
        raise CORAerror('CORA:wrongValue', 'first',
                       'Each variable should have minimum 2 levels')
    
    # Total number of design points
    total = np.prod(levels)
    
    # Initialization of output matrix
    des_mat = np.zeros((total, 0))
    
    # Loop for full factorial points
    for i in range(len(args)):
        if i != 0 and i != len(args) - 1:
            temp = np.zeros((0, 1))
            for j in range(np.prod(levels[:i])):
                temp1 = np.tile(varargin[i], (np.prod(levels[i+1:]), 1))
                temp1 = np.sort(temp1, axis=0)
                temp = np.vstack([temp, temp1])
        elif i == len(args) - 1:
            temp = np.tile(varargin[i], (total // levels[i], 1))
        else:
            temp = np.tile(varargin[i], (total // levels[i], 1))
            temp = np.sort(temp, axis=0)
        
        des_mat = np.hstack([des_mat, temp])
    
    return des_mat 