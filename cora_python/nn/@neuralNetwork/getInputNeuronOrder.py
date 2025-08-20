"""
getInputNeuronOrder - get input neuron order

Description:
    Get input neuron order

Syntax:
    neuronOrder = getInputNeuronOrder(method, x, inputSize)

Inputs:
    method - Method for ordering
    x - Input point
    inputSize - Input size specification

Outputs:
    neuronOrder - Neuron order

Example:
    neuronOrder = nn.getInputNeuronOrder('sensitivity', x, [10, 1, 1])

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       23-November-2022 (polish)
Last update:   23-November-2022 (polish)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import random
from typing import List

def getInputNeuronOrder(self, method: str, x: np.ndarray, inputSize: List[int]) -> List[int]:
    """
    Get input neuron order
    
    Args:
        method: Method for ordering
        x: Input point
        inputSize: Input size specification
        
    Returns:
        neuronOrder: Neuron order
    """
    # This method determines the order of input neurons based on the method
    if method == 'sensitivity':
        # Order by sensitivity magnitude
        if hasattr(self, 'sensitivity') and self.sensitivity is not None:
            # This would require computing sensitivity first
            pass
        return list(range(inputSize[0]))
    elif method == 'random':
        # Random ordering
        order = list(range(inputSize[0]))
        random.shuffle(order)
        return order
    else:
        # Default ordering
        return list(range(inputSize[0]))
