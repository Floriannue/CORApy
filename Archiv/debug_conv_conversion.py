"""
Debug Conv2D conversion
"""
import numpy as np
import sys
import os
sys.path.insert(0, 'cora_python')

from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.nn.neuralNetwork import NeuralNetwork

# Temporarily add debug prints to conversion
import cora_python.nn.neuralNetwork.convertDLToolboxNetwork as conv_module

# Monkey-patch to add debugging
original_aux_convertLayer = conv_module.aux_convertLayer

def debug_aux_convertLayer(layers, layer_dict, currentSize, verbose):
    layer_type = layer_dict.get('Type', 'Unknown')
    if layer_type == 'Conv2DLayer':
        print(f"\n=== DEBUG Conv2DLayer Conversion ===")
        print(f"  Layer dict keys: {layer_dict.keys()}")
        if 'Weight' in layer_dict:
            print(f"  Weight shape: {layer_dict['Weight'].shape}")
        if 'Bias' in layer_dict:
            print(f"  Bias shape: {layer_dict['Bias'].shape}")
            print(f"  Bias values (first 5): {layer_dict['Bias'].flatten()[:5]}")
        else:
            print(f"  Bias: NOT FOUND in layer_dict")
    
    result = original_aux_convertLayer(layers, layer_dict, currentSize, verbose)
    
    if layer_type == 'Conv2DLayer' and len(result[0]) > 0:
        last_layer = result[0][-1]
        if hasattr(last_layer, 'b'):
            print(f"  After conversion - b shape: {last_layer.b.shape}")
            print(f"  After conversion - b values (first 5): {last_layer.b.flatten()[:5]}")
    
    return result

conv_module.aux_convertLayer = debug_aux_convertLayer

# Now load the network
model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')

print(f"Loading model: {model_path}\n")
nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BCSS')

print(f"\n=== Final Network ===")
print(f"First layer type: {type(nn.layers[0]).__name__}")
if hasattr(nn.layers[0], 'b'):
    print(f"First layer b shape: {nn.layers[0].b.shape}")
    print(f"First layer b values (first 5): {nn.layers[0].b.flatten()[:5]}")

