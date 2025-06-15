"""
full_fact_mod - gives full factorial design matrix for any levels  
   more than 2 of any number of variables (minimum 2)

Syntax:
    des_mat = full_fact_mod(levels)

Inputs:
    levels - array of number of levels for each variable

Outputs:
    des_mat - mxn
          m = total number of designs (product of all levels) and  
          n = number of variables  
          The first column shows all the first variable levels,
          the second column shows second variable levels and so on. 

Example: 
    levels = [2, 3]  # 2 levels for first variable, 3 for second
    des_mat = full_fact_mod(levels)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors: Bhaskar Dongare, Matthias Althoff
         Python translation by AI Assistant
Written: 17-November-2008
Last update: 08-April-2016 (MA)
Last revision: ---
"""

import numpy as np
from typing import List, Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def full_fact_mod(levels: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    Gives full factorial design matrix for any levels more than 2 of any number of variables.
    
    Args:
        levels: Array of number of levels for each variable
        
    Returns:
        des_mat: mxn matrix where m = total number of designs (product of all levels) 
                and n = number of variables
                
    Raises:
        CORAerror: If any variable has less than 2 levels
    """
    levels = np.array(levels)
    
    if not np.all(levels > 1):
        raise CORAerror('CORA:wrongValue', 'first',
                       'Each variable should have minimum 2 levels')
    
    # Total number of design points
    total = np.prod(levels)
    
    # Initialization of output matrix
    des_mat = np.zeros((total, 0))
    
    # Loop for full factorial points
    for i in range(len(levels)):
        if i != 0 and i != len(levels) - 1:
            temp = np.zeros((0, 1))
            for j in range(np.prod(levels[:i])):
                temp1 = np.tile(np.arange(1, levels[i] + 1).reshape(-1, 1), 
                               (np.prod(levels[i+1:]), 1))
                temp1 = np.sort(temp1, axis=0)
                temp = np.vstack([temp, temp1])
        elif i == len(levels) - 1:
            temp = np.tile(np.arange(1, levels[i] + 1).reshape(-1, 1), 
                          (total // levels[i], 1))
        else:
            temp = np.tile(np.arange(1, levels[i] + 1).reshape(-1, 1), 
                          (total // levels[i], 1))
            temp = np.sort(temp, axis=0)
        
        des_mat = np.hstack([des_mat, temp])
    
    return des_mat 