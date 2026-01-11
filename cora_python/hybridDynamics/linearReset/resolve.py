"""
resolve - resolves inputs which are outputs from other equations

TRANSLATED FROM: cora_matlab/hybridDynamics/@linearReset/resolve.m

Syntax:
    linReset_ = resolve(linReset, flowList, stateBinds, inputBinds)

Inputs:
    linReset - linearReset object
    flowList - list of dynamical equations containing output equations
    stateBinds - states of the high-dimensional space that correspond to
                 the states of the low-dimensional reset object
    inputBinds - connections of input to global input/outputs of other
                 components

Outputs:
    linReset_ - linearReset object

Example: 
    -

Authors:       Maximilian Perschl, Mark Wetzlinger (MATLAB)
Written:       04-April-2022 (MATLAB)
Last update:   ---
Last revision: 11-October-2024 (MW, refactored from other functions) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, List, Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .linearReset import LinearReset


def resolve(linReset: 'LinearReset', flowList: List, 
            stateBinds: List[Union[np.ndarray, List[int]]],
            inputBinds: List[Union[np.ndarray, List]]) -> 'LinearReset':
    """
    Resolves inputs which are outputs from other equations.
    
    Args:
        linReset: linearReset object
        flowList: list of dynamical equations (LinearSys objects) containing output equations
        stateBinds: list of state indices for each component (0-based)
        inputBinds: list of input bind arrays, each with shape (num_inputs, 2)
                   where first column is component index (0=global, >0=other component)
                   and second column is output/input index
    
    Returns:
        LinearReset object with resolved inputs
    """
    from .linearReset import LinearReset
    from cora_python.contDynamics.linearSys.linearSys import LinearSys
    
    # Only supports linear output functions for input resolution
    if not all(isinstance(sys, LinearSys) for sys in flowList):
        raise CORAerror('CORA:wrongValue', 'second',
                       'All output functions for input resolution must be linear.')
    
    # Convert stateBinds to numpy arrays if needed
    stateBinds = [np.array(sb) if not isinstance(sb, np.ndarray) else sb 
                  for sb in stateBinds]
    # Convert inputBinds to numpy arrays if needed
    inputBinds = [np.array(ib) if not isinstance(ib, np.ndarray) else ib 
                  for ib in inputBinds]
    
    # Loop over all inputs of synchronized reset, three possibilities:
    #   - the input is a dummy input (same as global...?)
    #   - the input is a global input
    #   - the input is an output of another component
    # in the last case, we have to resolve the equation, i.e., the first input
    # to the first component is the second output of the second component
    numComp = len(inputBinds)
    
    # Rewrite inputs (list = component, values = global indices of component's inputs)
    # Calculate cumulative input indices for each component
    idxCompInput = []
    cumsum = 0
    for i in range(numComp):
        start_idx = cumsum
        cumsum += inputBinds[i].shape[0]
        end_idx = cumsum
        idxCompInput.append(np.arange(start_idx, end_idx))
    
    # Read out reset matrices and vector
    A = linReset.A.copy()
    B = linReset.B.copy()
    c = linReset.c.copy()
    
    # Loop over all components
    for thisCompIdx in range(numComp):
        # Resolve input binds of i-th component
        for j in range(inputBinds[thisCompIdx].shape[0]):
            # Index of other component (if global input, index is 0)
            # inputBinds uses MATLAB 1-based convention: 0 = global, 1 = component 0, 2 = component 1, etc.
            # We convert to Python 0-based: 0 = global, component indices start at 0
            otherCompIdx_raw = int(inputBinds[thisCompIdx][j, 0])
            
            # Check if ith input of synchronized reset is linked to a global
            # input (or is a dummy input)
            if otherCompIdx_raw == 0:
                # Nothing to resolve...
                continue
            
            # Convert from MATLAB 1-based to Python 0-based
            # inputBinds: 1 = component 0, 2 = component 1, etc.
            otherCompIdx = otherCompIdx_raw - 1
            
            # Obtain output equation of other component
            otherSys = flowList[otherCompIdx]
            C = otherSys.C
            D = otherSys.D if hasattr(otherSys, 'D') and otherSys.D is not None else np.zeros((otherSys.nrOfOutputs, otherSys.nrOfInputs))
            q = otherSys.k if hasattr(otherSys, 'k') and otherSys.k is not None else np.zeros((otherSys.nrOfOutputs, 1))
            
            # Current input is *l*-th output of other component
            # Python uses 0-based indexing for outputs
            l_raw = int(inputBinds[thisCompIdx][j, 1])
            # Convert from 1-based (MATLAB) to 0-based (Python)
            l = l_raw - 1
            
            # Convert the output matrix to a matrix for consistency below
            if np.isscalar(C) and C == 1:
                C = np.eye(otherSys.nrOfDims)
            
            # Ensure C, D, q are proper arrays
            if C.ndim == 1:
                C = C.reshape(1, -1)
            if D.ndim == 1:
                D = D.reshape(1, -1)
            if q.ndim == 0:
                q = np.array([[q]])
            elif q.ndim == 1:
                q = q.reshape(-1, 1)
            
            # Insert output l-th output of other component k, i.e., y_k(l), for
            # the j-th input of component i, i.e., u_i(j):
            # B_i(:,j) * u_i(j)
            # = B_i(:,j) * y_k(l)
            # = B_i(:,j) * ( C_k(l,:) * x_k + D_k(l,:) * u_k + q_k(l) )
            # = B_i(:,j) * C_k(l,:) * x_k
            #   + B_i(:,j) * D_k(l,:) * u_k
            #   + B_i(:,j) * q_k(l)
            
            # Check for circular input sequences: the term D_k(l,:)*u_k may
            # point back to u_i(j), so we force D_k(l,jj) to be zero or u_k(jj)
            # to be a global input to prevent infinite loops
            for jj in range(D.shape[1]):
                if not (D[l, jj] == 0 or (jj < len(inputBinds[otherCompIdx]) and 
                                          inputBinds[otherCompIdx][jj, 0] == 0)):
                    raise CORAerror('CORA:notSupported',
                        'It is not allowed for the feedthrough matrix D '
                        'to point to inputs that are defined by the '
                        'output of other subsystems, since it would '
                        'otherwise be able to construct infinite loops!')
            
            # Read out part of (full) B matrix relevant for the current mapping
            # MATLAB: B_part = B(stateBinds{thisCompIdx},idxCompInput{thisCompIdx}(j));
            B_part = B[np.ix_(stateBinds[thisCompIdx], [idxCompInput[thisCompIdx][j]])]
            # Ensure B_part is a column vector (MATLAB returns column vector)
            if B_part.ndim == 2 and B_part.shape[1] == 1:
                B_part = B_part.flatten()  # Flatten to 1D for easier handling
            
            # Compute effect: BC affects A, BD affects B, Bq affects c
            # MATLAB: BC_j = B_part * C(l,:);
            # MATLAB: BD_j = B_part * D(l,:);
            # MATLAB: Bq_j = B_part * q(l);
            # B_part is a column vector, C(l,:) is a row vector
            C_row = C[l, :] if C.shape[0] > l else C[0, :]
            # B_part (column) * C_row (row) = outer product, but MATLAB does matrix multiplication
            # If B_part is (n,1) and C_row is (1,m), then B_part * C_row = (n,m) matrix
            if B_part.ndim == 1:
                B_part_col = B_part.reshape(-1, 1)
            else:
                B_part_col = B_part
            BC_j = B_part_col @ C_row.reshape(1, -1)  # (n_states, 1) @ (1, n_other_states) = (n_states, n_other_states)
            
            D_row = D[l, :] if D.shape[0] > l else D[0, :]
            BD_j = B_part_col @ D_row.reshape(1, -1)  # (n_states, 1) @ (1, n_other_inputs) = (n_states, n_other_inputs)
            
            # q(l) is a scalar in MATLAB
            q_val = q[l, 0] if q.shape[0] > l else (q[0, 0] if q.size > 0 else 0)
            Bq_j = B_part_col * q_val  # (n_states, 1) * scalar = (n_states, 1)
            
            # Update state matrix
            A[np.ix_(stateBinds[thisCompIdx], stateBinds[otherCompIdx])] += BC_j
            
            # Update input matrix by effect of re-mapped global inputs
            if len(idxCompInput[otherCompIdx]) > 0:
                # BD_j might need reshaping to match B dimensions
                if BD_j.shape[1] == len(idxCompInput[otherCompIdx]):
                    B[np.ix_(stateBinds[thisCompIdx], idxCompInput[otherCompIdx])] += BD_j
                else:
                    # Handle dimension mismatch
                    for idx, col_idx in enumerate(idxCompInput[otherCompIdx]):
                        if idx < BD_j.shape[1]:
                            B[stateBinds[thisCompIdx], col_idx] += BD_j[:, idx].flatten()
            
            # Update offset vector
            c[stateBinds[thisCompIdx], 0] += Bq_j.flatten()
    
    # We now remove all columns of non-global inputs, as the corresponding
    # inputs have been resolved
    allBinds = np.vstack(inputBinds)
    idxGlobalInput = allBinds[:, 0] == 0
    B_global = B[:, idxGlobalInput]  # Only columns for global inputs
    
    # Up until now, input binds to the same global input were treated as
    # separate columns in B, one for each component with that same global input
    # We now merge these separate columns into one per global input
    globalInputBinds = allBinds[idxGlobalInput]
    if len(globalInputBinds) > 0:
        globalInputIndices = globalInputBinds[:, 1]  # MATLAB 1-based indices
        numGlobalInputs = int(np.max(globalInputIndices))  # MATLAB 1-based max index
        # MATLAB uses 1-based indexing for global inputs
        # Create merged B matrix
        B_merged = np.zeros((B_global.shape[0], numGlobalInputs))
        for i in range(1, numGlobalInputs + 1):  # MATLAB 1-based global input index
            # Find which columns in B_global correspond to this global input
            col_mask = globalInputIndices == i
            col_indices = np.where(col_mask)[0]
            if len(col_indices) > 0:
                # Sum all columns corresponding to this global input
                B_merged[:, i - 1] = np.sum(B_global[:, col_indices], axis=1)  # Convert to 0-based
        B = B_merged
    else:
        # No global inputs, B should be empty or have dummy input
        B = B_global
        if B.shape[1] == 0:
            # Constructor will add dummy input if needed
            pass
    
    # Initialize reset function (note that the constructor adds a dummy input
    # if there is none, so that we don't have to do this here)
    linReset_ = LinearReset(A, B, c)
    
    return linReset_

