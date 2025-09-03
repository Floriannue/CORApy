"""
backprop - compute the backpropagation for the previous input

Syntax:
    grad_in = nn.backprop(grad_out, options, idxLayer)

Inputs:
    nn - neuralNetwork
    grad_out - gradient of the output of the neural network
    options - training parameters
    idxLayer - indices of layers that should be evaluated

Outputs:
    grad_in - gradient w.r.t the input

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/evaluate, nnOptimizer

Authors:       Tobias Ladner, Lukas Koller
Written:       01-March-2023
Last update:   03-May-2023 (LK, added backprop for polyZonotope)
                25-May-2023 (LK, added options as function parameter)
                31-July-2023 (LK, return update gradients)
                04-August-2023 (LK, added layer indices)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from .neuralNetwork import NeuralNetwork
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def backprop(nn: NeuralNetwork, grad_out: Any, options: Optional[Dict[str, Any]] = None, 
            idxLayer: Optional[List[int]] = None) -> Any:
    """
    Compute the backpropagation for the previous input
    
    Args:
        nn: neuralNetwork object
        grad_out: gradient of the output of the neural network
        options: training parameters
        idxLayer: indices of layers that should be evaluated (defaults to all layers)
        
    Returns:
        grad_in: gradient w.r.t the input
    """
    # parse input
    if options is None:
        options = {}
    if idxLayer is None:
        idxLayer = list(range(len(nn.layers)))  # 0-based indexing like Python
    
    # validate input
    if not hasattr(nn, '__class__') or 'neuralNetwork' not in str(nn.__class__):
        raise ValueError("First argument must be a neuralNetwork object")
    
    # execute -----------------------------------------------------------------
    
    if isinstance(grad_out, (int, float, np.ndarray)):
        # numeric
        for i in reversed(idxLayer):
            layer_i = nn.layers[i]  # Convert to 0-based indexing
            # Retrieve stored input
            if options.get('nn', {}).get('train', {}).get('backprop', False):
                if hasattr(layer_i, 'backprop') and 'input' in layer_i.backprop.get('store', {}):
                    input_data = layer_i.backprop['store']['input']
                else:
                    # If no stored input, we need to handle this case
                    # For now, we'll raise an error as this indicates missing functionality
                    raise CORAerror('CORA:notSupported', 'No stored input found for backpropagation')
            else:
                # If backprop is not enabled, we need input data
                # This is a limitation of the current implementation
                raise CORAerror('CORA:notSupported', 'Backpropagation requires stored input data')
            
            grad_out = layer_i.backpropNumeric(input_data, grad_out, options)
        
        grad_in = grad_out
    else:
        raise CORAerror('CORA:notSupported', 
                       f'Set representation {type(grad_out).__name__} is not supported.')
    
    return grad_in
