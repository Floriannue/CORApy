"""
Neural Network Module

This module provides neural network functionality for CORA.
"""

# Import the main neural network class with all methods attached
from .neuralNetwork import NeuralNetwork

# Import layer classes
from .layers.nnLayer import nnLayer
from .layers.linear.nnLinearLayer import nnLinearLayer
from .layers.nonlinear.nnActivationLayer import nnActivationLayer
from .layers.nonlinear.nnReLULayer import nnReLULayer
from .layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer

__all__ = [
    'NeuralNetwork',
    'nnLayer',
    'nnLinearLayer', 
    'nnActivationLayer',
    'nnReLULayer',
    'nnLeakyReLULayer'
]
