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
        Constructor for neural network
        
        Args:
            layers: List of neural network layers
            name: Network name
        """
        self.layers = layers
        self.name = name
        
        # Initialize properties
        self.neurons_in = 0
        self.neurons_out = 0
        self.neurons = 0
        self.connections = 0
        self.inputSize = None
        self.options = {}
        
        # Initialize layer properties
        if layers:
            # Get input neurons from first layer
            if hasattr(layers[0], 'neurons_in'):
                self.neurons_in = layers[0].neurons_in
            
            # Get output neurons from last layer
            if hasattr(layers[-1], 'neurons_out'):
                self.neurons_out = layers[-1].neurons_out
            
            # Calculate total neurons and connections
            for layer in layers:
                if hasattr(layer, 'neurons_out'):
                    self.neurons += layer.neurons_out
                if hasattr(layer, 'W') and hasattr(layer, 'b'):
                    self.connections += layer.W.size + layer.b.size

