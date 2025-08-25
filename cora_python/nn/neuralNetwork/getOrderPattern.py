"""
getOrderPattern - returns the current order pattern

Syntax:
    pattern = getOrderPattern(obj)

Inputs:
    obj - object of class neuralNetwork

Outputs:
    pattern - cell array

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/refine

Authors:       Tobias Ladner
Written:       26-August-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, List


def getOrderPattern(obj: Any) -> List:
    """
    Return the current order pattern.
    
    Args:
        obj: NeuralNetwork object
        
    Returns:
        List of order patterns
    """
    pattern = []
    
    refinable_layers = obj.getRefinableLayers()
    for i in range(len(refinable_layers)):
        pattern.append(refinable_layers[i].order.T)  # Transpose like MATLAB
    
    return pattern
