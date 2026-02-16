"""test_reduce_adaptive_comparison - Test reduce('adaptive') implementation"""

import numpy as np
from cora_python.contSet.zonotope import Zonotope

# Test case from MATLAB example
c = np.array([[1], [0]])
G = np.array([[1, 3, 2, -1, 0.03, 0.02, -0.1], 
              [2, 0, -1, 1, 0.02, -0.01, 0.2]])
Z = Zonotope(c, G)

print("Testing reduce('adaptive') implementation")
print(f"Original Z: {Z.generators().shape[1]} generators")

# Test with different diagpercent values
for diagpercent in [0.1, 0.05, 0.01]:
    try:
        Z_red, dH, gredIdx = Z.reduce('adaptive', diagpercent)
        print(f"\ndiagpercent={diagpercent}:")
        print(f"  Reduced generators: {Z_red.generators().shape[1]}")
        print(f"  dHerror: {dH:.6e}")
        print(f"  gredIdx length: {len(gredIdx)}")
        print(f"  Reduced generators kept: {Z_red.generators().shape[1] - len(gredIdx)}")
    except Exception as e:
        print(f"\ndiagpercent={diagpercent}: ERROR - {e}")
        import traceback
        traceback.print_exc()

print("\nTest complete")
