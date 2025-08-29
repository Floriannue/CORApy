"""
neuralNetwork - object constructor for neural networks

Description:
    This class represents neural network objects with various layer types.

Syntax:
    obj = NeuralNetwork(layers)

Inputs:
    layers - list of neural network layers

Outputs:
    obj - generated neural network object

Example:
    from .layers.linear.nnLinearLayer import nnLinearLayer
    from .layers.nonlinear.nnReLULayer import nnReLULayer
    
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnReLULayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       23-November-2022 (polish)
Last update:   23-November-2022 (polish)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import List, Any, Optional, Tuple, Dict, Union
import time
import json
import random

class NeuralNetwork:
    """
    Neural network class
    
    This class represents neural network objects with various layer types.
    
    Properties:
        layers: List of neural network layers
        neurons_in: Number of input neurons
        neurons_out: Number of output neurons
        neurons: Total number of neurons
        connections: Number of connections
        name: Network name
        inputSize: Input size specification
        options: Network options
    """
    
    def __init__(self, layers: List[Any], name: str = "Neural Network"):
        """
        Constructor for neural network (matches MATLAB exactly)
        
        Args:
            layers: List of neural network layers
            name: Network name
        """
        # Check inputs (matches MATLAB narginchk(0,1))
        if layers is None:
            layers = []
        
        # Validate layers (matches MATLAB validation)
        if not isinstance(layers, list):
            raise ValueError('First argument should be a list of type nnLayer.')
        
        # Check that all layers are nnLayer instances
        for i, layer in enumerate(layers):
            print(f"DEBUG: Layer {i}: {type(layer)}, has __class__: {hasattr(layer, '__class__')}, has type: {hasattr(layer, 'type')}")
            if not hasattr(layer, '__class__') or not hasattr(layer, 'type'):
                raise ValueError(f'Layer {i} must be nnLayer instance. Got: {type(layer)}')
        
        self.layers = layers
        self.name = name
        
        # Initialize properties (matches MATLAB exactly)
        self.neurons_in = None
        self.neurons_out = None
        self.reductionRate = 1  # matches MATLAB property
        
        # Simple neurons_in and _out computation (matches MATLAB exactly)
        print(f"DEBUG: Computing neurons_in from {len(self.layers)} layers")
        for i in range(len(self.layers)):
            try:
                nin, _ = self.layers[i].getNumNeurons()
                print(f"DEBUG: Layer {i} getNumNeurons() returned: nin={nin}")
                if nin is not None and nin != []:  # Check for non-None and non-empty list
                    self.neurons_in = nin
                    print(f"DEBUG: Set neurons_in = {nin}")
                    break
            except Exception as e:
                print(f"DEBUG: Layer {i} getNumNeurons() failed: {e}")
                continue
        
        print(f"DEBUG: Computing neurons_out from {len(self.layers)} layers")
        for i in range(len(self.layers) - 1, -1, -1):
            try:
                _, nout = self.layers[i].getNumNeurons()
                print(f"DEBUG: Layer {i} getNumNeurons() returned: nout={nout}")
                if nout is not None and nout != []:  # Check for non-None and non-empty list
                    self.neurons_out = nout
                    print(f"DEBUG: Set neurons_out = {nout}")
                    break
            except Exception as e:
                print(f"DEBUG: Layer {i} getNumNeurons() failed: {e}")
                continue
        

        self.setInputSize()

    
    def __len__(self) -> int:
        """Returns the number of layers"""
        return len(self.layers)
    
    def __repr__(self) -> str:
        """String representation of the neural network"""
        return f"NeuralNetwork(name='{self.name}', layers={len(self.layers)}, neurons_in={self.neurons_in}, neurons_out={self.neurons_out})"
    
    def __str__(self) -> str:
        """String representation of the neural network"""
        return self.__repr__()
    


