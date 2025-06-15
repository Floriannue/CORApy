"""
block_zeros - initializes a cell array for a number of blocks with
   all-zero column vectors of block dimension; if a single block is
   given, we instead return a vector instead of a cell

Syntax:
    S_out = block_zeros(blocks)

Inputs:
    blocks - bx2 array with b blocks

Outputs:
    S_out - (cell array of) all-zero vector(s) of block dimension

Example:
    S_out = block_zeros([[1, 2], [3, 5]])

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger
         Python translation by AI Assistant
Written: 16-October-2024
Last update: ---
Last revision: ---
"""

import numpy as np
from typing import Union, List


def block_zeros(blocks: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Initializes a cell array for a number of blocks with all-zero column vectors.
    
    Args:
        blocks: bx2 array with b blocks, where each row defines [start, end] indices
        
    Returns:
        S_out: If single block, returns zero vector. Otherwise, returns list of zero vectors.
    """
    blocks = np.array(blocks)
    if blocks.ndim == 1:
        blocks = blocks.reshape(1, -1)
    
    numBlocks = blocks.shape[0]
    
    if numBlocks == 1:
        # Single block case - return vector
        S_out = np.zeros((blocks[-1, -1], 1))
    else:
        # Multiple blocks case - return list of vectors
        S_out = []
        for i in range(numBlocks):
            block_size = blocks[i, 1] - blocks[i, 0] + 1
            S_out.append(np.zeros((block_size, 1)))
    
    return S_out 