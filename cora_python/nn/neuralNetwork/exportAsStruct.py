"""
exportAsStruct - exports this network as a struct

Syntax:
    nn = nn.exportAsStruct()

Inputs:
    nn - neuralNetwork

Outputs:
    struct - struct representation of the network

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/importFromStruct

Authors:       Tobias Ladner
Written:       10-November-2023
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Dict, List
from .neuralNetwork import NeuralNetwork


def exportAsStruct(nn: NeuralNetwork) -> Dict[str, Any]:
    """
    Exports this network as a struct
    
    Args:
        nn: neuralNetwork object
        
    Returns:
        struct: struct representation of the network
    """
    # Basic network information
    nnStruct = {
        'type': 'neuralNetwork',
        'neurons_in': getattr(nn, 'neurons_in', None),
        'neurons_out': getattr(nn, 'neurons_out', None),
        'layers': []
    }
    
    # Export each layer
    for i, layer in enumerate(nn.layers):
        layer_info = {
            'index': i,
            'type': layer.__class__.__name__
        }
        
        # Add layer-specific information
        if hasattr(layer, 'exportAsStruct'):
            layer_struct = layer.exportAsStruct()
            layer_info.update(layer_struct)
        else:
            # Basic layer info
            if hasattr(layer, 'name'):
                layer_info['name'] = layer.name
            if hasattr(layer, 'inputSize'):
                layer_info['inputSize'] = layer.inputSize
        
        nnStruct['layers'].append(layer_info)
    
    return nnStruct
