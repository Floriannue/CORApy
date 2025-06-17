"""
block_operation - apply operation to block-decomposed sets

This function applies a binary operation to block-decomposed sets.

Syntax:
    result = block_operation(operation, sets1, sets2)

Inputs:
    operation - function to apply (e.g., enclose, plus)
    sets1 - single set or list of sets (block-decomposed)
    sets2 - single set or list of sets (block-decomposed)

Outputs:
    result - single set or list of sets after operation

Authors: Python translation by AI Assistant
Written: 2025
"""

from typing import Union, List, Callable


def block_operation(operation: Callable, sets1: Union[object, List[object]], 
                   sets2: Union[object, List[object]] = None):
    """
    Apply operation to block-decomposed sets
    
    Args:
        operation: Function to apply (e.g., enclose, plus, center)
        sets1: Single set or list of sets (block-decomposed)
        sets2: Single set or list of sets (block-decomposed), optional for unary operations
        
    Returns:
        Single set or list of sets after operation
    """
    # Handle unary operations (when sets2 is None)
    if sets2 is None:
        if not isinstance(sets1, list):
            return operation(sets1)
        else:
            # Apply operation to each block
            result = []
            for set1 in sets1:
                result_block = operation(set1)
                result.append(result_block)
            
            # If there's only one block, return the set directly
            if len(result) == 1:
                return result[0]
            return result
    
    # If both are single sets (not decomposed), perform regular operation
    if not isinstance(sets1, list) and not isinstance(sets2, list):
        return operation(sets1, sets2)
    
    # If one is a list and the other is not, convert the single set to a list
    if not isinstance(sets1, list):
        sets1 = [sets1]
    if not isinstance(sets2, list):
        sets2 = [sets2]
    
    # Check that both lists have the same length
    if len(sets1) != len(sets2):
        raise ValueError(f"Block lists must have same length: {len(sets1)} vs {len(sets2)}")
    
    # Apply operation to each corresponding pair of blocks
    result = []
    for set1, set2 in zip(sets1, sets2):
        result_block = operation(set1, set2)
        result.append(result_block)
    
    # If there's only one block, return the set directly
    if len(result) == 1:
        return result[0]
    
    return result 