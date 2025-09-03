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
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def getInputNeuronOrder(self, method: str, x: np.ndarray, inputSize=None) -> np.ndarray:
    """
    Get input neuron order
    
    Args:
        method: Method for ordering ('in-order', 'sensitivity', 'snake')
        x: Input point
        inputSize: Input size specification (optional)
        
    Returns:
        neuronOrder: Neuron order (1-based indexing like MATLAB)
    """
    # determine order based on method
    if method == 'in-order':
        # just iterate through all pixels
        if inputSize is None or len(inputSize) == 0:
            neuronOrder = np.arange(1, x.size + 1)  # 1-based indexing
        else:
            neuronOrder = np.arange(1, inputSize[0] * inputSize[1] + 1)  # 1-based indexing
    
    elif method == 'snake':
        # spiral inwards - MATLAB style snake pattern
        if inputSize is None or len(inputSize) < 2:
            raise CORAerror('CORA:wrongValue', "Snake method requires inputSize with at least 2 dimensions")
        
        # Create 1-based indexing array
        featOrderNaive = np.arange(1, inputSize[0] * inputSize[1] + 1)
        A = featOrderNaive.reshape(inputSize[0], inputSize[1])
        
        neuronOrder = []
        while A.size > 0:
            # Add first column (top to bottom)
            neuronOrder.extend(A[:, 0].tolist())
            # Remove first column
            if A.shape[1] > 1:
                A = A[:, 1:]
            else:
                break
            
            # Add last row (left to right)
            if A.size > 0:
                neuronOrder.extend(A[-1, :].tolist())
                # Remove last row
                if A.shape[0] > 1:
                    A = A[:-1, :]
                else:
                    break
            
            # Add last column (bottom to top)
            if A.size > 0:
                neuronOrder.extend(A[:, -1][::-1].tolist())
                # Remove last column
                if A.shape[1] > 1:
                    A = A[:, :-1]
                else:
                    break
            
            # Add first row (right to left)
            if A.size > 0:
                neuronOrder.extend(A[0, :][::-1].tolist())
                # Remove first row
                if A.shape[0] > 1:
                    A = A[1:, :]
                else:
                    break
    
    elif method == 'sensitivity':
        # identify least sensitive input neurons
        S, _ = self.calcSensitivity(x)
        

        # mean across output neurons (axis=1 in MATLAB corresponds to axis=1 in Python)
        # S has shape (nK, nK, bSz) where nK is output neurons, bSz is batch size
        S = np.mean(np.abs(S), axis=1)  # This gives us (nK, bSz)
        
        # For single input, we need to take the mean across the batch dimension
        if S.ndim > 1 and S.shape[1] > 1:
            S = np.mean(S, axis=1)  # This gives us (nK,) - one value per input neuron
        
        if inputSize is not None and len(inputSize) > 0:
            # mean across channels
            S = S.reshape(inputSize)
            S = np.mean(S, axis=2)
            S = S.reshape(-1, 1)
        
        # Sort by sensitivity (ascending order) and convert to 1-based indexing
        neuronOrder = np.argsort(S.flatten()) + 1  # 1-based indexing
    
    else:
        raise CORAerror('CORA:wrongValue', f"Unknown method: {method}")
    
    return np.array(neuronOrder)
