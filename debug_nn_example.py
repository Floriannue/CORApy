#!/usr/bin/env python3
"""
Debug script for neural network examples
"""

import sys
import os
import numpy as np

# Add the cora_python path to sys.path
sys.path.insert(0, os.path.abspath('.'))

print("=== Testing Neural Network Example Imports ===")

try:
    print("1. Testing basic imports...")
    from cora_python.nn.neuralNetwork import NeuralNetwork
    print("✓ Successfully imported NeuralNetwork")
except ImportError as e:
    print(f"✗ Failed to import NeuralNetwork: {e}")
    sys.exit(1)

try:
    print("2. Testing vnnlib2cora import...")
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    print("✓ Successfully imported vnnlib2cora")
except ImportError as e:
    print(f"✗ Failed to import vnnlib2cora: {e}")
    sys.exit(1)

try:
    print("3. Testing validateNNoptions import...")
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    print("✓ Successfully imported validateNNoptions")
except ImportError as e:
    print(f"✗ Failed to import validateNNoptions: {e}")
    sys.exit(1)

try:
    print("4. Testing layer imports...")
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    print("✓ Successfully imported layer classes")
except ImportError as e:
    print(f"✗ Failed to import layer classes: {e}")
    sys.exit(1)

print("\n=== Testing Neural Network Construction ===")

try:
    print("5. Testing neural network construction...")
    # Create simple layers
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnReLULayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)
    print("✓ Successfully created NeuralNetwork")
    print(f"  - neurons_in: {nn.neurons_in}")
    print(f"  - neurons_out: {nn.neurons_out}")
    print(f"  - layers: {len(nn.layers)}")
except Exception as e:
    print(f"✗ Failed to create NeuralNetwork: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Testing VNNLIB Reading ===")

try:
    print("6. Testing VNNLIB file reading...")
    specPath = "cora_python/examples/nn/models/prop_1.vnnlib"
    if os.path.exists(specPath):
        X0, specs = vnnlib2cora(specPath)
        print("✓ Successfully read VNNLIB file")
        print(f"  - X0 length: {len(X0)}")
        print(f"  - specs type: {specs.type}")
        if X0:
            print(f"  - X0[0] type: {type(X0[0])}")
            if hasattr(X0[0], 'inf') and hasattr(X0[0], 'sup'):
                print(f"  - X0[0].inf shape: {X0[0].inf.shape}")
                print(f"  - X0[0].sup shape: {X0[0].sup.shape}")
    else:
        print(f"⚠ VNNLIB file not found: {specPath}")
except Exception as e:
    print(f"✗ Failed to read VNNLIB file: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing ONNX Network Reading ===")

try:
    print("7. Testing ONNX network reading...")
    modelPath = "cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx"
    if os.path.exists(modelPath):
        nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BSSC')
        print("✓ Successfully read ONNX network")
        print(f"  - neurons_in: {nn.neurons_in}")
        print(f"  - neurons_out: {nn.neurons_out}")
        print(f"  - layers: {len(nn.layers)}")
    else:
        print(f"⚠ ONNX file not found: {modelPath}")
except Exception as e:
    print(f"✗ Failed to read ONNX network: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing Options Validation ===")

try:
    print("8. Testing options validation...")
    options = {}
    options['nn'] = {
        'use_approx_error': True,
        'poly_method': 'bounds',
        'num_generators': 100,
        'train': {
            'backprop': False,
            'mini_batch_size': 512
        }
    }
    
    validated_options = validateNNoptions(options, True)
    print("✓ Successfully validated options")
    print(f"  - poly_method: {validated_options['nn']['poly_method']}")
    print(f"  - num_generators: {validated_options['nn']['num_generators']}")
except Exception as e:
    print(f"✗ Failed to validate options: {e}")
    import traceback
    traceback.print_exc()

print("\n=== All Tests Completed ===")
