import numpy as np
import sys
sys.path.insert(0, '.')

from cora_python.contSet.zonotope import Zonotope

# Simulate the test case
N = 100
dim_x = 2 * N  # 200

# Create C matrix like in the test
C = np.zeros((1, 2 * N))  # shape (1, 200)
C[0, 0] = 1

# Create zonotope like R0 in the test
R0 = Zonotope(np.zeros((dim_x, 1)), np.zeros((dim_x, 0)))
print(f"C shape: {C.shape}")
print(f"R0.c shape: {R0.c.shape}")
print(f"R0.G shape: {R0.G.shape}")
print(f"R0.dim(): {R0.dim()}")

# Try the multiplication
try:
    result = C @ R0
    print(f"Success! Result.c shape: {result.c.shape}, Result.G shape: {result.G.shape}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
