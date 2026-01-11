"""
synchronize - synchronize linear reset functions of equal pre-state and
   post-state dimensions by adding and concatenating their respective
   matrices, which corresponds to the reset functions being evaluated
   synchronously

TRANSLATED FROM: cora_matlab/hybridDynamics/@linearReset/synchronize.m

Syntax:
    linReset_sync = synchronize(linResets)
    linReset_sync = synchronize(linResets, idStates)

Inputs:
    linResets - list/array of linearReset objects
    idStates - indices of states that are mapped by identity (optional)

Outputs:
    linReset_sync - linearReset object

Example: 
    -

Authors:       Maximilian Perschl, Mark Wetzlinger (MATLAB)
Written:       04-April-2022 (MATLAB)
Last update:   01-July-2022 (MATLAB)
                14-January-2023 (MW, handle states unaffected by sync) (MATLAB)
Last revision: 10-October-2024 (MW, moved from transition class) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, List, Union, Optional
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .linearReset import LinearReset


def synchronize(linResets: List['LinearReset'], 
                idStates: Optional[Union[np.ndarray, List[int]]] = None) -> 'LinearReset':
    """
    Synchronize linear reset functions of equal pre-state and post-state dimensions.
    
    Args:
        linResets: list of linearReset objects
        idStates: optional indices of states that are mapped by identity (0-based)
                 Can be a boolean array, list of indices, or numpy array
    
    Returns:
        LinearReset object with synchronized reset functions
    """
    from .linearReset import LinearReset
    
    if len(linResets) == 0:
        raise CORAerror('CORA:wrongValue', 'first', 'At least one reset function is required.')
    
    # Set default value for identity mapping
    preStateDim = linResets[0].preStateDim
    postStateDim = linResets[0].postStateDim
    if idStates is None:
        idStates = []
    
    # All reset functions need to have the same mapping
    preStateDims = [r.preStateDim for r in linResets]
    postStateDims = [r.postStateDim for r in linResets]
    if any(d != preStateDims[0] for d in preStateDims) or any(d != postStateDims[0] for d in postStateDims):
        raise CORAerror('CORA:wrongValue', 'first',
                       'All reset functions must map the same spaces: R^n -> R^m.')
    inputDim = linResets[0].inputDim
    
    # Initialize state/input matrix and constant offset
    A = np.zeros((postStateDim, preStateDim))
    B = np.zeros((postStateDim, inputDim))
    c = np.zeros((postStateDim, 1))
    
    # Since all resets have been projected to higher dimensions before,
    # we can just add them here
    for linReset in linResets:
        A = A + linReset.A
        B = B + linReset.B
        c = c + linReset.c
    
    # Insert identity mapping
    # MATLAB: id = zeros(postStateDim,1); id(idStates) = 1; A = A + diag(id);
    # idStates can be: empty [], logical array, or index array (0-based in Python)
    id = np.zeros((postStateDim, 1))
    
    if idStates is not None and (isinstance(idStates, (list, np.ndarray)) and len(idStates) > 0):
        # Convert to numpy array if needed
        if isinstance(idStates, list):
            idStates = np.array(idStates)
        
        # Handle boolean array (logical indexing like MATLAB)
        if isinstance(idStates, np.ndarray) and idStates.dtype == bool:
            # Boolean array: set positions where True to 1 (0-based indexing)
            if len(idStates) == postStateDim:
                id[idStates, 0] = 1
        else:
            # Index array - Python uses 0-based indexing
            idStates_arr = np.atleast_1d(idStates).flatten()
            if len(idStates_arr) > 0:
                # Ensure indices are within bounds (0-based)
                id_indices = idStates_arr[(idStates_arr >= 0) & (idStates_arr < postStateDim)]
                if len(id_indices) > 0:
                    id[id_indices, 0] = 1
    
    # Add diagonal matrix: A = A + diag(id)
    A = A + np.diag(id.flatten())
    
    # Init synchronized reset
    linReset_sync = LinearReset(A, B, c)
    
    return linReset_sync

