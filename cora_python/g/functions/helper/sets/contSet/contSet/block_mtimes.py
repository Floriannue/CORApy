"""
block_mtimes - block matrix multiplication for sets

This function performs matrix multiplication on block-decomposed sets.

Syntax:
    result = block_mtimes(matrix, sets)

Inputs:
    matrix - matrix to multiply with
    sets - single set or list of sets (block-decomposed)

Outputs:
    result - single set or list of sets after multiplication

Authors: Python translation by AI Assistant
Written: 2025
"""

import numpy as np
from typing import Union, List


def block_mtimes(matrix: np.ndarray, sets: Union[object, List[object]]):
    """
    Block matrix multiplication for sets
    
    Args:
        matrix: Matrix to multiply with
        sets: Single set or list of sets (block-decomposed)
        
    Returns:
        Single set or list of sets after multiplication
    """
    # If sets is a single set (not decomposed), perform regular multiplication
    if not isinstance(sets, list):
        # Regular matrix multiplication
        return matrix @ sets
    
    # If sets is a list (decomposed), we need to handle block operations
    # This implements the MATLAB logic: for each output block i, compute
    # S_i = sum_j M_ij * S_j where M_ij is the block from matrix
    
    num_blocks = len(sets)
    
    # Determine dimensions of each block from the sets
    if hasattr(sets[0], 'dim') and callable(sets[0].dim):
        dims = [s.dim() for s in sets]
    elif hasattr(sets[0], 'shape'):
        dims = [s.shape[0] for s in sets]
    else:
        # Fallback: assume each block has size determined by its content
        dims = [len(s) if hasattr(s, '__len__') else 1 for s in sets]
    
    # Calculate block boundaries (end indices)
    blocks_end = np.cumsum(dims)
    blocks = np.column_stack([
        np.concatenate([[0], blocks_end[:-1]]),  # start indices
        blocks_end - 1  # end indices (inclusive)
    ])
    
    # Initialize result list
    result = []
    
    # Formula: ∀ i: S_i = ∑_j M_ij * S_j
    for i in range(num_blocks):
        # Initialize with zeros
        result_block = None
        
        for j in range(num_blocks):
            # Extract matrix block M_ij
            row_start, row_end = blocks[i, 0], blocks[i, 1] + 1
            col_start, col_end = blocks[j, 0], blocks[j, 1] + 1
            M_block = matrix[row_start:row_end, col_start:col_end]
            
            # Compute M_ij * S_j
            term = M_block @ sets[j]
            
            # Add to result
            if result_block is None:
                result_block = term
            else:
                result_block = result_block + term
        
        result.append(result_block)
    
    # If there's only one block, return the set directly
    if len(result) == 1:
        return result[0]
    
    return result 