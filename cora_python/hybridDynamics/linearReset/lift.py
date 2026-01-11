"""
lift - lifts a linear reset function to a higher-dimensional space

TRANSLATED FROM: cora_matlab/hybridDynamics/@linearReset/lift.m

Syntax:
    linReset_ = lift(linReset, N, M, stateBind, inputBind, id)

Inputs:
    linReset - linearReset object
    N - dimension of the higher-dimensional state space
    M - dimension of higher-dimensional input space
    stateBind - states of the high-dimensional space that correspond to
               the states of the low-dimensional reset object
    inputBind - inputs of the high-dimensional space that correspond to
                the inputs of the low-dimensional reset object
    id - true/false whether identity reset function should be used for all
         other states

Outputs:
    linReset_ - lifted linearReset object

Example:
    A = [1 2; 0 -1]; B = [2 0 1; -1 0 0]; c = [1; -5];
    linReset = LinearReset(A, B, c);
    N = 6; stateBind = [2, 3];
    M = 5; inputBind = [2, 3, 4];
    id = True;
    linReset_ = lift(linReset, N, M, stateBind, inputBind, id);

Authors:       Mark Wetzlinger (MATLAB)
Written:       07-September-2024 (MATLAB)
Last update:   10-October-2024 (MW, support input matrix) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union, List
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .linearReset import LinearReset


def lift(linReset: 'LinearReset', N: int, M: int, 
         stateBind: Union[np.ndarray, List[int]], 
         inputBind: Union[np.ndarray, List[int]], 
         id: bool) -> 'LinearReset':
    """
    Lifts a linear reset function to a higher-dimensional space.
    
    Args:
        linReset: linearReset object
        N: dimension of the higher-dimensional state space
        M: dimension of higher-dimensional input space
        stateBind: indices of states in high-dimensional space (0-based in Python)
        inputBind: indices of inputs in high-dimensional space (0-based in Python)
        id: whether identity reset function should be used for other states
    
    Returns:
        LinearReset object in higher-dimensional space
    """
    from .linearReset import LinearReset
    
    # Note: reset function needs to map from R^n -> R^n
    if linReset.preStateDim != linReset.postStateDim:
        raise CORAerror('CORA:notSupported',
            'Projection of reset functions to higher-dimensional spaces '
            'only supported for R^n -> R^n.')
    
    # Convert to numpy arrays if needed
    if isinstance(stateBind, (list, int)):
        stateBind = np.array([stateBind] if isinstance(stateBind, int) else stateBind)
    elif not isinstance(stateBind, np.ndarray):
        stateBind = np.array(stateBind)
    
    if isinstance(inputBind, (list, int)):
        inputBind = np.array([inputBind] if isinstance(inputBind, int) else inputBind)
    elif not isinstance(inputBind, np.ndarray):
        inputBind = np.array(inputBind)
    
    # Ensure 1D arrays
    stateBind = np.atleast_1d(stateBind).flatten()
    inputBind = np.atleast_1d(inputBind).flatten()
    
    # All indices are 0-based (Python convention)
    stateBind_py = stateBind
    inputBind_py = inputBind
    
    # Ensure indices are within bounds
    if np.any(stateBind_py < 0) or np.any(stateBind_py >= N):
        raise CORAerror('CORA:wrongValue', 'stateBind', 
                       'State bind indices must be in range [0, N-1]')
    if np.any(inputBind_py < 0) or np.any(inputBind_py >= M):
        raise CORAerror('CORA:wrongValue', 'inputBind',
                       'Input bind indices must be in range [0, M-1]')
    
    # Init A matrix by identity or zeros
    if id:
        Aproj = np.eye(N)
    else:
        Aproj = np.zeros((N, N))
    Bproj = np.zeros((N, M))
    cProj = np.zeros((N, 1))
    
    # Insert state and input mappings A and B, and vector c
    # Use advanced indexing to set the submatrices
    # stateBind_py is 0-based indices
    Aproj[np.ix_(stateBind_py, stateBind_py)] = linReset.A
    Bproj[np.ix_(stateBind_py, inputBind_py)] = linReset.B
    cProj[stateBind_py] = linReset.c
    
    # Construct resulting reset object
    linReset_ = LinearReset(Aproj, Bproj, cProj)
    
    return linReset_

