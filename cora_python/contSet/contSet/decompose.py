"""
decompose - block decomposition of a set into a cell array of projected
    sets (no cell if only one block)

Syntax:
    S_out = decompose(S,blocks)

Inputs:
    S - contSet object
    blocks - bx2 array with b blocks

Outputs:
    S_out - list of contSet objects

Example:
    S = zonotope([1;-1;0;0;1],eye(5));
    blocks = [[1, 2], [3, 5]];
    S_out = decompose(S,blocks);

References:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: priv_reach_decomp

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       16-October-2024
Last update:   ---
Last revision: ---
"""

import numpy as np
from .project import project

def decompose(S, blocks):
    """
    Block decomposition of a set into projected sets.
    
    Args:
        S: contSet object
        blocks: numpy array of shape (b, 2) where each row defines a block [start, end]
        
    Returns:
        list: List of projected contSet objects, or single object if only one block
    """
    # Convert blocks to numpy array if needed
    blocks = np.asarray(blocks)
    
    # Number of blocks
    num_blocks = blocks.shape[0]
    
    # No projection if a single block
    if num_blocks == 1:
        # Return a copy of the original set
        if hasattr(S, 'copy'):
            return S.copy()
        else:
            # Fallback: try to create a copy manually
            return type(S)(S)
    
    # Use project function for each block
    S_out = []
    for i in range(num_blocks):
        # Create range from start to end (inclusive, converting to 0-based indexing)
        start = blocks[i, 0] - 1  # Convert from 1-based to 0-based
        end = blocks[i, 1]        # End is inclusive in MATLAB style
        dims = list(range(start, end))
        
        # Project the set to the specified dimensions
        projected_set = project(S, dims)
        S_out.append(projected_set)
    
    return S_out 