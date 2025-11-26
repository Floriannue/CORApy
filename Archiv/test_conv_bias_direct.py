import sys
sys.path.insert(0, 'cora_python')

import numpy as np
from nn.layers.linear.nnConv2DLayer import nnConv2DLayer

# Create a simple Conv2D layer with bias
W = np.random.randn(2, 2, 1, 32)  # [H, W, in_c, out_c]
b = np.random.randn(32)  # 32 output channels
padding = np.array([0, 0, 0, 0])
stride = np.array([1, 1])
dilation = np.array([1, 1])

print(f"Creating Conv2D layer with:")
print(f"  W shape: {W.shape}")
print(f"  b shape: {b.shape}, size: {b.size}")

try:
    layer = nnConv2DLayer(W, b, padding, stride, dilation, name='test')
    print(f"\nLayer created successfully!")
    print(f"  self.b shape: {layer.b.shape}, size: {layer.b.size}")
    print(f"  self.W shape: {layer.W.shape}")
except Exception as e:
    print(f"\nError creating layer: {e}")
    import traceback
    traceback.print_exc()

