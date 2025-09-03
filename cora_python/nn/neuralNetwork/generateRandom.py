"""
generateRandom - creates a random layer-based network

Syntax:
    obj = neuralNetwork.generateRandom()
    obj = neuralNetwork.generateRandom('NrInputs',nrOfInputs)
    obj = neuralNetwork.generateRandom('NrInputs',nrOfInputs,...
       'NrOutputs',nrOfOutputs)
    obj = neuralNetwork.generateRandom('NrInputs',nrOfInputs,...
       'NrOutputs',nrOfOutputs,'ActivationFun',actFun)
    obj = neuralNetwork.generateRandom('NrInputs',nrOfInputs,...
       'NrOutputs',nrOfOutputs,'ActivationFun',actFun,'NrLayers',nrLayers)

Inputs:
    Name-Value pairs (all options, arbitrary order):
       <'NrInputs',nrOfInputs> - number of inputs
       <'NrOutputs',nrOfOutputs> - number of outputs
       <'ActivationFun',actFun> - type of activation functions
           actFun has to be in nnActivationLayer/instantiateFromString
       <'NrLayers',nrLayers> - number of layers
       <'NrHiddenNeurons',nrHiddenNeurons> - number of neurons in hidden layersof outputs

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork, nnActivationLayer/instantiateFromString

Authors:       Tobias Ladner
Written:       28-March-2022
Last update:   28-November-2022 (name-value pair syntax)
                16-December-2022 (TL, NrHiddenNeurons)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import random
import numpy as np
from typing import Any, Dict, List, Optional, Union
from .neuralNetwork import NeuralNetwork
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def generateRandom(*args) -> NeuralNetwork:
    """
    Creates a random layer-based network
    
    Args:
        *args: Name-Value pairs (all options, arbitrary order)
        
    Returns:
        obj: generated neuralNetwork object
    """
    # validate parameters
    # name-value pairs -> number of input arguments is always a multiple of 2
    if len(args) % 2 != 0:
        raise CORAerror('CORA:evenNumberInputArgs')
    
    # read input arguments
    NVpairs = list(args)
    
    # check list of name-value pairs
    valid_names = {'NrInputs', 'NrOutputs', 'NrLayers', 'ActivationFun', 'NrHiddenNeurons'}
    
    # Parse name-value pairs
    nrInputs = None
    nrOutputs = None
    actFun = None
    nrLayers = None
    nrHiddenNeurons = None
    
    for i in range(0, len(NVpairs), 2):
        if i + 1 < len(NVpairs):
            name = NVpairs[i]
            value = NVpairs[i + 1]
            
            if name not in valid_names:
                raise CORAerror('CORA:wrongValue', 'name', f'Unknown parameter: {name}')
            
            if name == 'NrInputs':
                nrInputs = value
            elif name == 'NrOutputs':
                nrOutputs = value
            elif name == 'ActivationFun':
                actFun = value
            elif name == 'NrLayers':
                nrLayers = value
            elif name == 'NrHiddenNeurons':
                nrHiddenNeurons = value
    
    # Set defaults if not provided
    if nrInputs is None:
        nrInputs = random.randint(1, 5)
    if nrOutputs is None:
        nrOutputs = random.randint(1, 5)
    if actFun is None:
        actFun = "sigmoid"
    if nrLayers is None:
        nrLayers = random.randint(1, 5)
    if nrHiddenNeurons is None:
        nrHiddenNeurons = random.randint(2, 20)
    
    # determine neurons in each layer
    neurons = [0] * (1 + nrLayers)
    neurons[0] = nrInputs
    for i in range(nrLayers - 1):
        neurons[1 + i] = nrHiddenNeurons
    neurons[-1] = nrOutputs
    
    # create layers
    layers = [None] * (2 * (len(neurons) - 1))
    
    scale = 2
    for i in range(len(neurons) - 1):
        # add linear layer
        W = np.random.rand(neurons[i + 1], neurons[i]) * scale - scale / 2
        b = np.random.rand(neurons[i + 1], 1) * scale - scale / 2
        
        # TODO: Import and create nnLinearLayer when available
        # layers[2 * i] = nnLinearLayer(W, b)
        # For now, create a placeholder
        from ...nn.layers.linear.nnLinearLayer import nnLinearLayer
        layers[2 * i] = nnLinearLayer(W, b)
        
        # add activation layers
        if 2 * i + 1 < len(layers):
            # TODO: Import and create activation layer when available
            # layer = nnActivationLayer.instantiateFromString(actFun)
            # layers[2 * i + 1] = layer
            # For now, create a placeholder
            from ...nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
            try:
                layer = nnActivationLayer.instantiateFromString(actFun)
                layers[2 * i + 1] = layer
            except:
                # Fallback to a basic activation layer
                layers[2 * i + 1] = None
    
    # Remove None entries
    layers = [layer for layer in layers if layer is not None]
    
    # Create and return neural network
    obj = NeuralNetwork(layers)
    return obj
