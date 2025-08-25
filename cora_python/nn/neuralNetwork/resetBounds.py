"""
resetBounds - resets the bounds of all layers

Syntax:
    resetBounds(obj)

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


def resetBounds(obj: NeuralNetwork) -> None:
    """
    Resets the bounds of all layers
    
    Args:
        obj: neuralNetwork object
    """
    for layer in obj.layers:
        if hasattr(layer, 'l'):
            layer.l = []
        if hasattr(layer, 'u'):
            layer.u = []
