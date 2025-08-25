"""
importFromStruct - imports a network from struct

Syntax:
    nn = neuralNetwork.importFromStruct(nnStruct)

Inputs:
    nnStruct - struct representation of the network

Outputs:
    nn - neuralNetwork

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/exportAsStruct

Authors:       Tobias Ladner
Written:       10-November-2023
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Dict, List
from .neuralNetwork import NeuralNetwork
import numpy as np


def importFromStruct(nnStruct: Dict[str, Any]) -> NeuralNetwork:
    """
    Imports a network from struct
    
    Args:
        nnStruct: struct representation of the network
        
    Returns:
        nn: neuralNetwork
    """
    # Basic validation
    if not isinstance(nnStruct, dict):
        raise ValueError("nnStruct must be a dictionary")
    
    if 'type' not in nnStruct or nnStruct['type'] != 'neuralNetwork':
        raise ValueError("nnStruct must have type 'neuralNetwork'")
    
    # Create empty neural network
    layers = []
    
    # Import layers if available
    if 'layers' in nnStruct:
        for layer_info in nnStruct['layers']:
            if isinstance(layer_info, dict) and 'type' in layer_info:
                layer_type = layer_info['type']
                
                # Create layer based on type
                if layer_type == 'nnLinearLayer':
                    from ...nn.layers.linear.nnLinearLayer import nnLinearLayer
                    W = layer_info.get('W', np.eye(1))
                    b = layer_info.get('b', np.zeros((1, 1)))
                    layer = nnLinearLayer(W, b)
                    if 'name' in layer_info:
                        layer.name = layer_info['name']
                    layers.append(layer)
                    
                elif layer_type == 'nnElementwiseAffineLayer':
                    from ...nn.layers.linear.nnElementwiseAffineLayer import nnElementwiseAffineLayer
                    scale = layer_info.get('scale', 1.0)
                    offset = layer_info.get('offset', 0.0)
                    layer = nnElementwiseAffineLayer(scale, offset)
                    if 'name' in layer_info:
                        layer.name = layer_info['name']
                    layers.append(layer)
                    
                elif layer_type == 'nnIdentityLayer':
                    from ...nn.layers.linear.nnIdentityLayer import nnIdentityLayer
                    layer = nnIdentityLayer()
                    if 'name' in layer_info:
                        layer.name = layer_info['name']
                    layers.append(layer)
                    
                elif layer_type == 'nnReshapeLayer':
                    from ..layers.other.nnReshapeLayer import nnReshapeLayer
                    idx_out = layer_info.get('idx_out', [1])
                    layer = nnReshapeLayer(idx_out)
                    if 'name' in layer_info:
                        layer.name = layer_info['name']
                    layers.append(layer)
                    
                elif layer_type == 'nnActivationLayer':
                    # For activation layers, we need to determine the specific type
                    activation_type = layer_info.get('activation', 'relu')
                    from ...nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
                    try:
                        layer = nnActivationLayer.instantiateFromString(activation_type)
                        if 'name' in layer_info:
                            layer.name = layer_info['name']
                        layers.append(layer)
                    except Exception:
                        # Fallback to a basic activation layer if specific type not available
                        from ...nn.layers.nonlinear.nnReLULayer import nnReLULayer
                        layer = nnReLULayer()
                        if 'name' in layer_info:
                            layer.name = layer_info['name']
                        layers.append(layer)
                        
                else:
                    # For unknown layer types, create a placeholder but log a warning
                    print(f"Warning: Unknown layer type '{layer_type}' - creating placeholder")
                    layers.append(None)
    
    # Create neural network
    nn = NeuralNetwork(layers)
    
    # Set additional attributes if they exist
    if 'neurons_in' in nnStruct:
        nn.neurons_in = nnStruct['neurons_in']
    if 'neurons_out' in nnStruct:
        nn.neurons_out = nnStruct['neurons_out']
    
    return nn
