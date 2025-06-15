"""
sparseOrthMatrix - generates a sparse orthogonal matrix

Syntax:
    Q = sparseOrthMatrix(n)

Inputs:
    n - dimension

Outputs:
    Q - sparse orthogonal matrix

Example:
    n = 5
    Q = sparseOrthMatrix(n)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger
         Python translation by AI Assistant
Written: 07-October-2019
Last update: ---
Last revision: ---
"""

import numpy as np
from typing import Union
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def sparseOrthMatrix(n: int) -> np.ndarray:
    """
    Generates a sparse orthogonal matrix.
    
    Args:
        n: Dimension of the matrix
        
    Returns:
        Q: Sparse orthogonal matrix of size n x n
        
    Raises:
        CORAerror: If resulting matrix is not orthogonal
    """
    # Generate blocks
    min_block = 2
    max_block = 4
    remaining = n
    i = 0
    block = []
    
    while True:
        if i == 0:
            # Fix one block to size 2
            if n <= 4:
                block.append(n)
                break
            next_size = 2
        else:
            next_size = min_block + int(np.floor(np.random.rand() * (max_block - min_block + 1)))
        
        if (remaining - next_size == 0 or 
            (remaining >= next_size and remaining - next_size >= min_block)):
            block.append(next_size)
            i += 1
            remaining = remaining - next_size
        
        if remaining == 0:
            break
    
    # Write Q in block structure
    blocks = len(block)
    Q = np.zeros((n, n))
    curr_start = 0
    curr_end = 0
    
    for i in range(blocks):
        # Generate random matrix and get QR decomposition
        random_mat = 2 * np.random.rand(block[i], block[i]) - 1
        Qblock, _ = np.linalg.qr(random_mat)
        curr_end = curr_end + block[i]
        Q[curr_start:curr_end, curr_start:curr_end] = Qblock
        curr_start = curr_start + block[i]
    
    # Reorder columns and rows -> no more strict block structure
    reOrderCols = np.random.permutation(n)
    reOrderRows = np.random.permutation(n)
    Q = Q[reOrderRows, :]
    Q = Q[:, reOrderCols]
    
    # Reorder such that last row has two non-zero entries
    if np.count_nonzero(Q[-1, :]) != 2:
        order = np.arange(n)
        for i in range(n - 1):
            if np.count_nonzero(Q[i, :]) == 2:
                order[i] = n - 1
                order[-1] = i
                Q = Q[order, :]
                break
    
    # Check if Q really orthogonal
    det_Q = np.abs(np.linalg.det(Q))
    col_norms = np.linalg.norm(Q, axis=0)
    
    if not withinTol(det_Q, 1, 1e-9) or not np.all(withinTol(col_norms, 1, 1e-9)):
        raise CORAerror('CORA:specialError', 'Resulting matrix is not orthogonal')
    
    return Q 