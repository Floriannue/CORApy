"""
resetApproxOrder - resets the approximation order of all layers

Syntax:
    resetApproxOrder(obj)

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


def resetApproxOrder(obj: NeuralNetwork) -> None:
    """
    Resets the approximation order of all layers
    
    Args:
        obj: neuralNetwork object
    """
    for layer in obj.layers:
        if hasattr(layer, 'order'):
            layer.order = [1]  # Reset to default order
        if hasattr(layer, 'do_refinement'):
            layer.do_refinement = True
