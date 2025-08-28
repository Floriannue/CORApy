"""
evaluateZonotopeBatch_ - evaluate neural network for a batch of zonotopes
  without setting default options.

Syntax:
    [c, G] = nn.evaluateZonotopeBatch_(c, G, options, idxLayer)

Inputs:
    c, G - batch of zonotope; [n,q+1,b] = size([c G]),
       where n is the number of dims, q the number of generators, and b the batch size
    options - parameter for neural network evaluation
    idxLayer - indices of layers that should be evaluated

Outputs:
    c, G - batch of output sets

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/evaluate, neuralNetwork/prepareForZonoBatchEval

Authors:       Lukas Koller
Written:       02-August-2023
Last update:   08-August-2023 (moved code to layers)
               22-February-2024 (merged options.nn, moved input storage handling from layer to network)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from .neuralNetwork import NeuralNetwork


def evaluateZonotopeBatch_(nn: NeuralNetwork, c: np.ndarray, G: np.ndarray, 
                          options: Dict[str, Any], idxLayer: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate neural network for a batch of zonotopes without setting default options.
    
    Args:
        nn: NeuralNetwork object
        c, G: batch of zonotope; [n,q+1,b] = size([c G]),
           where n is the number of dims, q the number of generators, and b the batch size
        options: parameter for neural network evaluation
        idxLayer: indices of layers that should be evaluated
        
    Returns:
        c, G: batch of output sets
    """
    for i in idxLayer:
        layeri = nn.layers[i]
        # Store input for backpropagation
        if options.get('nn', {}).get('train', {}).get('backprop', False):
            if hasattr(layeri, 'backprop') and hasattr(layeri.backprop, 'store'):
                layeri.backprop.store.inc = c
                layeri.backprop.store.inG = G
        
        c, G = layeri.evaluateZonotopeBatch(c, G, options)
    
    return c, G
