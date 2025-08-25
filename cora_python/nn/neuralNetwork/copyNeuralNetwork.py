"""
copyNeuralNetwork - Create a copy of the neural network.

Syntax:
    [nnCopy] = copyNeuralNetwork(obj)

Inputs:
    obj - neural network

Outputs:
    nnCopy - copy of the neural network obj
    
References:---

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Lukas Koller
Written:       21-June-2023
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import copy
from typing import Any
from .neuralNetwork import NeuralNetwork


def copyNeuralNetwork(obj: NeuralNetwork) -> NeuralNetwork:
    """
    Create a copy of the neural network.
    
    Args:
        obj: neural network
        
    Returns:
        nnCopy: copy of the neural network obj
    """
    # copy layers of the neural network
    K = len(obj.layers)
    layers = [None] * K
    
    for k in range(K):
        # equivalent to layer.copy() in MATLAB
        if hasattr(obj.layers[k], 'copy'):
            layers[k] = obj.layers[k].copy()
        else:
            # fallback to deep copy if no copy method available
            layers[k] = copy.deepcopy(obj.layers[k])
    
    nnCopy = NeuralNetwork(layers)
    
    return nnCopy
