"""
castWeights - cast learnable weights of a neural network

Syntax:
    nn.castWeights(x, idxLayer)

Inputs:
    nn - object of class neuralNetwork
    x - instance of data type
    idxLayer - indices of layers that should be evaluated

Outputs:
    -

References:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Lukas Koller
Written:       04-December-2023
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List, Optional, Union
from .neuralNetwork import NeuralNetwork


def castWeights(nn: NeuralNetwork, x: np.ndarray, idxLayer: Optional[List[int]] = None) -> None:
    """
    Cast learnable weights of a neural network
    
    Args:
        nn: NeuralNetwork object
        x: instance of data type
        idxLayer: indices of layers that should be evaluated (defaults to all layers)
    """
    # validate parameters
    if idxLayer is None:
        # 1-based indexing like MATLAB
        idxLayer = list(range(len(nn.layers)))  # 0-based indexing like Python
    
    for i in idxLayer:
        layeri = nn.layers[i]  # Convert to 0-based indexing
        # move all learnable parameters to gpu
        names = layeri.getLearnableParamNames()
        for j in range(len(names)):
            # cast learnable weights
            if hasattr(layeri, names[j]):
                current_value = getattr(layeri, names[j])
                # Cast to same type as x
                if hasattr(x, 'dtype'):
                    setattr(layeri, names[j], current_value.astype(x.dtype))
                else:
                    # If x is not numpy array, try to cast to its type
                    setattr(layeri, names[j], type(x)(current_value))
        
        # Notify layer
        if hasattr(layeri, 'castWeights'):
            layeri.castWeights(x)
