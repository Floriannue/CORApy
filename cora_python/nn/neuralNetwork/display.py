"""
display - displays the properties of a neuralNetwork object

Syntax:
    display(obj)

Inputs:
    obj - neuralNetwork object

Outputs:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner
Written:       23-November-2022
Last update:   17-January-2023 (TL, better layer output)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any
from .neuralNetwork import NeuralNetwork


def display(obj: NeuralNetwork) -> str:
    """
    Displays the properties of a neuralNetwork object
    
    Args:
        obj: NeuralNetwork object
        
    Returns:
        str: display string
    """
    # Build display string
    display_str = ""
    
    # Network class info
    display_str += f"Neural network: '{obj.__class__.__name__}'\n"
    
    # in/out neurons
    if hasattr(obj, 'neurons_in') and obj.neurons_in is not None:
        display_str += f"Nr. of input neurons: {obj.neurons_in}\n"
    else:
        display_str += "Nr. of input neurons: unknown\n"
        
    if hasattr(obj, 'neurons_out') and obj.neurons_out is not None:
        display_str += f"Nr. of output neurons: {obj.neurons_out}\n"
    else:
        display_str += "Nr. of output neurons: unknown\n"
    
    display_str += "\n"
    
    # Layers info
    display_str += f"layers: ({len(obj.layers)} layers)\n"
    for i in range(len(obj.layers)):
        layer_i = obj.layers[i]
        if hasattr(layer_i, 'getLayerInfo'):
            layer_info = layer_i.getLayerInfo()
        else:
            layer_info = str(layer_i.__class__.__name__)
        display_str += f" ({i+1})\t {layer_info}\n"
    
    display_str += "\n"
    
    # Print to console (like MATLAB's display)
    print(display_str)
    
    return display_str
