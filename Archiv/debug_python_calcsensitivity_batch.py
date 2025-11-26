"""Debug script to check calcSensitivity behavior with batch inputs"""
import numpy as np
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir)
sys.path.insert(0, project_root)

from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions

# Load the network
modelPath = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx'
nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BSSC')

# Create options
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
options = validateNNoptions(options, True)
options['nn']['interval_center'] = False

# Test with single batch
print("=== Single batch ===")
xi_single = np.array([[0.67985776], [0.5], [0.5], [0.5], [-0.45]], dtype=np.float32)
print(f"xi_single shape: {xi_single.shape}")

S_single, y_single = nn.calcSensitivity(xi_single, options, store_sensitivity=False)
print(f"S_single shape: {S_single.shape}")
print(f"y_single shape: {y_single.shape}")

# Test with multiple batches
print("\n=== Multiple batches (5) ===")
xi_multi = np.tile(xi_single, (1, 5))
print(f"xi_multi shape: {xi_multi.shape}")

S_multi, y_multi = nn.calcSensitivity(xi_multi, options, store_sensitivity=False)
print(f"S_multi shape: {S_multi.shape}")
print(f"y_multi shape: {y_multi.shape}")

# Test with large batch (512)
print("\n=== Large batch (512) ===")
xi_large = np.tile(xi_single, (1, 512))
print(f"xi_large shape: {xi_large.shape}")

S_large, y_large = nn.calcSensitivity(xi_large, options, store_sensitivity=False)
print(f"S_large shape: {S_large.shape}")
print(f"y_large shape: {y_large.shape}")

# Compute sens from S_large
print("\n=== Computing sens from S_large ===")
S_abs = np.abs(S_large)
S_max = np.maximum(S_abs, 1e-6)
sens_max = np.max(S_max, axis=0)
print(f"sens_max shape: {sens_max.shape}")
sens = sens_max.T
print(f"sens shape: {sens.shape}")
print(f"Expected: (512, 5)")

