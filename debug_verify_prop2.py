"""
Debug script to understand why prop_2 counterexample isn't detected
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora

# Enable debug mode
debug_options = {
    'nn': {
        'use_approx_error': True,
        'poly_method': 'bounds',
        'train': {
            'backprop': False,
            'mini_batch_size': 512
        },
        '_debug_verify': True  # Enable debug output
    }
}

cora_root = CORAROOT()
modelPath = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx')
specPath = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'prop_2.vnnlib')

if not os.path.isfile(modelPath) or not os.path.isfile(specPath):
    print(f"Files not found: modelPath={modelPath}, specPath={specPath}")
    sys.exit(1)

# Read network and specs (matching test file)
from cora_python.tests.nn.neuralNetwork.test_neuralNetwork_verify import aux_readModelAndSpecs
nn, x, r, A, b, safeSet, options = aux_readModelAndSpecs(modelPath, specPath)
# Update options with debug settings
if 'nn' not in options:
    options['nn'] = {}
options['nn'].update(debug_options['nn'])

print(f"safeSet = {safeSet}")
print(f"A shape: {A.shape}")
print(f"b shape: {b.shape}")
print(f"x shape: {x.shape}")
print(f"r shape: {r.shape}")

# Run verification with debug output
print("\n" + "="*80)
print("Running verify with debug output...")
print("="*80 + "\n")

res, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout=10, verbose=True)

print(f"\n" + "="*80)
print(f"Result: {res}")
print(f"x_ = {x_}")
print(f"y_ = {y_}")
print("="*80)

