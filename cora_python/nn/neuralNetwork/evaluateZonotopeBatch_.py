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
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from .neuralNetwork import NeuralNetwork


def evaluateZonotopeBatch_(nn: NeuralNetwork, c, G, 
                          options: Dict[str, Any], idxLayer: List[int]):
    """
    Evaluate neural network for a batch of zonotopes without setting default options.
    
    Args:
        nn: NeuralNetwork object
        c, G: batch of zonotope; [n,q+1,b] = size([c G]),
           where n is the number of dims, q the number of generators, and b the batch size
           (numpy arrays or torch tensors) - converted to torch internally
        options: parameter for neural network evaluation
        idxLayer: indices of layers that should be evaluated
        
    Returns:
        c, G: batch of output sets (torch tensors)
    """
    # Convert numpy inputs to torch if needed
    if isinstance(c, np.ndarray):
        c = torch.tensor(c, dtype=torch.float32)
    if isinstance(G, np.ndarray):
        G = torch.tensor(G, dtype=torch.float32)
    # Validate layer indices
    num_layers = len(nn.layers)
    for idx in idxLayer:
        if idx < 0 or idx >= num_layers:
            raise IndexError(f"Layer index {idx} out of bounds for network with {num_layers} layers")

    for i in idxLayer:
        layeri = nn.layers[i]
        # Store input for backpropagation
        if options.get('nn', {}).get('train', {}).get('backprop', False):
            if not hasattr(layeri, 'backprop') or not isinstance(layeri.backprop, dict):
                layeri.backprop = {'store': {}}
            if 'store' not in layeri.backprop or not isinstance(layeri.backprop['store'], dict):
                layeri.backprop['store'] = {}
            layeri.backprop['store']['inc'] = c
            layeri.backprop['store']['inG'] = G
        
        c, G = layeri.evaluateZonotopeBatch(c, G, options)
    
    return c, G
