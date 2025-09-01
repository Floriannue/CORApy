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

def getInputNeuronOrder(self, method: str, x: np.ndarray, inputSize=None) -> List[int]:
    """
    Get input neuron order
    
    Args:
        method: Method for ordering ('in-order', 'sensitivity', 'snake')
        x: Input point
        inputSize: Input size specification (optional)
        
    Returns:
        neuronOrder: Neuron order
    """
    # determine order based on method
    if method == 'in-order':
        # just iterate through all pixels
        if inputSize is None or len(inputSize) == 0:
            neuronOrder = list(range(x.size))
        else:
            neuronOrder = list(range(inputSize[0] * inputSize[1]))
    
    elif method == 'snake':
        # spiral inwards
        if inputSize is None or len(inputSize) < 2:
            raise ValueError("Snake method requires inputSize with at least 2 dimensions")
        
        featOrderNaive = list(range(1, inputSize[0] * inputSize[1] + 1))
        A = np.array(featOrderNaive).reshape(inputSize[0], inputSize[1])
        
        neuronOrder = []
        while A.size > 0:
            neuronOrder.extend(A[0, :].tolist())
            A = np.fliplr(A[1:, :]).T
    
    elif method == 'sensitivity':
        # identify least sensitive input neurons
        S, _ = self.calcSensitivity(x)
        
        # mean across output neurons
        S = np.mean(np.abs(S), axis=0)
        if inputSize is not None and len(inputSize) > 0:
            # mean across channels
            S = S.reshape(inputSize)
            S = np.mean(S, axis=2)
            S = S.reshape(-1, 1)
        
        # Sort by sensitivity (ascending order)
        neuronOrder = np.argsort(S.flatten()).tolist()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return neuronOrder
