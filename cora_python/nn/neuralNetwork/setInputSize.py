"""
setInputSize - propagate inputSize through the network and store the
inputSize for each layer. This is necessary to propagate images through a
network.

Syntax:
    outputSize = setInputSize(obj, inputSize)

Inputs:
    inputSize - column vector, with sizes of each dimension
    verbose: bool if information should be displayed

Outputs:
    outputSize - output size of the neural network

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: NeuralNetwork

Authors:       Lukas Koller, Tobias Ladner
Written:       10-June-2022
Last update:   17-January-2023 (TL, polish)
Last revision: 17-August-2022
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List, Optional, Union
from .neuralNetwork import NeuralNetwork
from ...g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def setInputSize(obj: NeuralNetwork, inputSize: Optional[List[int]] = None, 
                verbose: bool = False) -> List[int]:
    """
    Propagate inputSize through the network and store the inputSize for each layer
    
    Args:
        obj: neuralNetwork object
        inputSize: column vector, with sizes of each dimension
        verbose: bool if information should be displayed
        
    Returns:
        outputSize: output size of the neural network
    """
    # parse input
    if inputSize is None:
        if not hasattr(obj, 'neurons_in') or obj.neurons_in is None:
            raise CORAerror("CORA:specialError", 
                          "Please provide an input size. Unable to determine it from network weights.")
        inputSize = [obj.neurons_in, 1]
    
    # compute in-/out sizes of all layers
    if verbose:
        print("Computing in-/out sizes of all layers...")
    
    obj.neurons_in = np.prod(inputSize)
    
    for i in range(len(obj.layers)):
        # iterate through all layers
        layer_i = obj.layers[i]
        outputSize = layer_i.computeSizes(inputSize)
        if verbose:
            print(f" ({i+1})\t {layer_i.getLayerInfo()}")
        inputSize = outputSize
    
    obj.neurons_out = np.prod(inputSize)
    
    return inputSize
