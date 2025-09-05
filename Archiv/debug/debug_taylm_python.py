# Debug script to test Taylm empty creation in Python
import numpy as np
from cora_python.contSet.taylm.taylm import Taylm

print("Python Results:")

# Test empty creation
n = 2
empty_tay = Taylm.empty(n)

print(f'Created empty Taylm with n={n}')
print(f'Type: {type(empty_tay)}')
print(f'Dim: {empty_tay.dim()}')
print(f'Is empty: {empty_tay.isemptyobject()}')

# Check monomials structure
print(f'Monomials type: {type(empty_tay.monomials)}')
if hasattr(empty_tay.monomials, 'shape'):
    print(f'Monomials shape: {empty_tay.monomials.shape}')
else:
    print(f'Monomials length: {len(empty_tay.monomials)}')

# Test with different dimensions
for n in [0, 1, 3, 5]:
    empty_tay = Taylm.empty(n)
    print(f'n={n}: dim={empty_tay.dim()}, isempty={empty_tay.isemptyobject()}')

# Check what the empty function actually creates
print("\nDebugging empty function:")
monomials = np.zeros((0, 2))
coefficients = np.array([])
remainder = np.array([0, 0])
print(f"monomials shape: {monomials.shape}")
print(f"coefficients shape: {coefficients.shape}")
print(f"remainder shape: {remainder.shape}")

# Debug the dim function directly
print("\nDebugging dim function:")
print(f"monomials.shape[1]: {monomials.shape[1]}")
print(f"len(monomials.shape): {len(monomials.shape)}")
print(f"monomials.shape[1] > 0: {monomials.shape[1] > 0}")
print(f"len(monomials.shape) == 2: {len(monomials.shape) == 2}")
print(f"Both conditions: {len(monomials.shape) == 2 and monomials.shape[1] > 0}")
