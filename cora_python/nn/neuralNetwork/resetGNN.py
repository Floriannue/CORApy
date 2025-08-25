"""
resetGNN - resets the GNN-specific properties of all layers

Syntax:
    resetGNN(obj)

Inputs:
    obj - object of class neuralNetwork

Outputs:
    None

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Tobias Ladner
Written:       30-November-2022
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any
from .neuralNetwork import NeuralNetwork


def resetGNN(obj: NeuralNetwork) -> None:
    """
    Resets the GNN-specific properties of all layers
    
    Args:
        obj: neuralNetwork object
    """
    for layer in obj.layers:
        if hasattr(layer, 'merged_neurons'):
            layer.merged_neurons = []
