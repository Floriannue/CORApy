"""
reset - resets the neural network by deleting all values in all
   layers used for internal computations

Syntax:
    reset(obj)

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


def reset(obj: NeuralNetwork) -> None:
    """
    Resets the neural network by deleting all values in all layers used for internal computations
    
    Args:
        obj: neuralNetwork object
    """
    if hasattr(obj, 'resetApproxOrder'):
        obj.resetApproxOrder()
    if hasattr(obj, 'resetBounds'):
        obj.resetBounds()
    if hasattr(obj, 'resetGNN'):
        obj.resetGNN()
