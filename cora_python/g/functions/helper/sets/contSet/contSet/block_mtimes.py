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
    # For now, implement a simple version that applies the matrix to each block
    # This is a simplified implementation - the full version would handle
    # proper block matrix operations
    
    result = []
    for i, set_block in enumerate(sets):
        # Apply matrix multiplication to each block
        # Note: This assumes the matrix is compatible with each block
        result_block = matrix @ set_block
        result.append(result_block)
    
    # If there's only one block, return the set directly
    if len(result) == 1:
        return result[0]
    
    return result 