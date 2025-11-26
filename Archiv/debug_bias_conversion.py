import sys
sys.path.insert(0, 'cora_python')

from nn.neuralNetwork.readONNXNetwork import aux_readONNXviaPython
from nn.neuralNetwork.convertDLToolboxNetwork import convertDLToolboxNetwork

# Read the ONNX model (same as test_conv_input.py)
import os
from cora_python.g.macros.CORAROOT import CORAROOT
model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')
print(f"Reading ONNX model: {model_path}")
intermediate_net = aux_readONNXviaPython(model_path, 'BCSS', 'BC', 'dagnetwork')
layer_dicts = intermediate_net['Layers']

print(f"\nNumber of layers read: {len(layer_dicts)}")

# Find Conv2D layers and print their bias info
for i, layer_dict in enumerate(layer_dicts):
    print(f"\nLayer {i}: {layer_dict.get('Type', 'Unknown')}")
    if layer_dict.get('Type') == 'Conv2DLayer':
        print(f"  Name: {layer_dict.get('Name', 'N/A')}")
        W = layer_dict.get('Weight')
        b = layer_dict.get('Bias')
        print(f"  Weight shape: {W.shape if W is not None else 'None'}")
        print(f"  Bias shape: {b.shape if hasattr(b, 'shape') else type(b)}")
        if hasattr(b, 'shape'):
            print(f"  Bias size: {b.size}")
            if b.size <= 10:
                print(f"  Bias values: {b}")

print("\n" + "="*60)
print("Now converting to CORA network...")
print("="*60)

nn = convertDLToolboxNetwork(layer_dicts, verbose=True)

print(f"\nNetwork created successfully!")
print(f"  Input size: {nn.neurons_in}")
print(f"  Output size: {nn.neurons_out}")
print(f"  Number of layers: {nn.number_of_layers}")

