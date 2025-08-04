"""
batchCombinator - Function which generates a sampling without
   repetitions/replacement of the set 1:N, taken K at a time.
   Can be called batch-wise to enable large output.
   Based on the combinator by Matt Fig.

Authors: Michael Eichelbeck (MATLAB)
         Automatic python translation: Florian Nüssel BA 2025
Written: 27-July-2022 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


def batchCombinator(N: int, K: int, batch_size: int, state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate combinations in batches to handle large cases efficiently.
    
    Args:
        N: sample set size
        K: sample drawn at a time. Requirement: K<=N
        batch_size: number of combinations to be returned
        state: dict with combinator state (None for initialization)
        
    Returns:
        CN: generated combinations
        state: new combinator state
    """
    
    if state is None or 'BC' not in state:  # init
        if K > N:
            state = {'done': True}
            return np.array([]), state
        
        state = {
            'M': float(N),  # Single will give us trouble on indexing
            'WV': list(range(1, K + 1)),  # Working vector
            'lim': K,  # Sets the limit for working index
            'inc': 1,  # Controls which element of WV is being worked on
            'stp': 0,  # internal tracker
            'flg': 0,  # internal tracker
            'BC': int(np.floor(np.prod(range(N - K + 1, N + 1)) / np.prod(range(1, K + 1)))),  # total number of combinations
            'ii': 1,  # global index
            'done': False  # have all combinations been returned?
        }
    
    # copy dict fields to tmp variables for speed
    M = state['M']
    WV = state['WV'].copy()
    lim = state['lim']
    inc = state['inc']
    stp = state['stp']
    flg = state['flg']
    BC = state['BC']
    ii = state['ii']
    done = state['done']
    
    n = batch_size
    
    if ii + n > BC:  # last batch
        n = BC - (ii - 1)
        n = int(n)
    
    CN = np.zeros((n, K), dtype=int)
    
    for nii in range(n):
        if ii == BC:  # The last row
            CN[nii, :] = list(range(N - K + 1, N + 1))
            done = True
        elif ii == 1:  # The first row
            CN[nii, :] = WV
        else:  # all other rows
            if (inc + lim) > N:
                stp = inc  # This is where the for loop below stops
                flg = 0  # Used for resetting inc
            else:
                stp = 1
                flg = 1
            
            for jj in range(stp):
                WV[K + jj - inc] = lim + jj + 1  # Faster than a vector assignment
            
            CN[nii, :] = WV  # Make assignment
            inc = inc * flg + 1  # Increment the counter
            lim = WV[K - inc]  # lim for next run
        
        ii = ii + 1
    
    # copy local variables to state dict
    state['M'] = M
    state['WV'] = WV
    state['lim'] = lim
    state['inc'] = inc
    state['stp'] = stp
    state['flg'] = flg
    state['BC'] = BC
    state['ii'] = ii
    state['done'] = done
    
    return CN, state 