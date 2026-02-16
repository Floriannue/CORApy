"""compare_quadmap_and_hessian - Compare quadMap implementation and H values"""

import numpy as np
from cora_python.contSet.zonotope import Zonotope

print("=" * 80)
print("COMPARING quadMap IMPLEMENTATION")
print("=" * 80)

# Test quadMap with a simple case
print("\n1. Testing quadMap with simple inputs:")

# Create a simple zonotope
Z = Zonotope(np.array([[1.0], [0.0]]), np.array([[1.0, 0.5], [0.0, 0.3]]))

# Create a simple Hessian (2x2x2 tensor)
# H[i] is the Hessian matrix for dimension i
H = [
    np.array([[0.1, 0.05], [0.05, 0.1]]),  # H[0]
    np.array([[0.2, 0.1], [0.1, 0.2]])     # H[1]
]

print(f"Z center: {Z.center().flatten()}")
print(f"Z generators shape: {Z.generators().shape}")
print(f"H[0] shape: {H[0].shape}, H[1] shape: {H[1].shape}")

try:
    errorSec = 0.5 * Z.quadMap(H)
    print(f"\nquadMap result:")
    print(f"  Center: {errorSec.center().flatten()}")
    print(f"  Generators shape: {errorSec.generators().shape}")
    print(f"  Radius: {np.sum(np.abs(errorSec.generators()), axis=1)}")
    print(f"  Radius max: {np.max(np.sum(np.abs(errorSec.generators()), axis=1))}")
except Exception as e:
    print(f"ERROR in quadMap: {e}")
    import traceback
    traceback.print_exc()

# Compare with MATLAB formula
print("\n2. Checking quadMap formula:")
print("MATLAB formula (from aux_quadMapSingle):")
print("  c = 0.5 * sum(diag(H{i} * G * G')) for each i")
print("  G_new = [H{i} * G for each i]")
print("\nPython should match this formula exactly")

# Read Python quadMap implementation
print("\n3. Python quadMap implementation check:")
with open('cora_python/contSet/zonotope/quadMap.py', 'r') as f:
    lines = f.readlines()
    # Find the key computation
    for i, line in enumerate(lines):
        if 'c =' in line and 'H' in line:
            print(f"  Line {i+1}: {line.strip()}")
        if 'G_new' in line or 'generators' in line:
            if i < len(lines) - 1 and ('H' in line or 'H' in lines[i+1]):
                print(f"  Line {i+1}: {line.strip()}")

print("\n" + "=" * 80)
print("NEXT: Compare H (Hessian) values between Python and MATLAB")
print("=" * 80)
