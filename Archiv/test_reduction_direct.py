"""test_reduction_direct - Direct test of reduction algorithm"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.private.priv_reduceAdaptive import priv_reduceAdaptive

print("=" * 80)
print("DIRECT TEST OF REDUCTION ALGORITHM")
print("=" * 80)

# Create test zonotope
# Use values that might cause different behavior
np.random.seed(42)  # For reproducibility

# Create a 2D zonotope with 5 generators
center = np.array([[0.1], [0.1]])
generators = np.array([
    [0.05, 0.03, 0.02, 0.01, 0.005],
    [0.03, 0.05, 0.01, 0.02, 0.003]
])

Z = Zonotope(center, generators)
print(f"\nOriginal Z:")
print(f"  Center: {center.flatten()}")
print(f"  Generators shape: {generators.shape}")
print(f"  Number of generators: {generators.shape[1]}")

# Test with different diagpercent values
diagpercent_values = [0.1, 0.2, 0.3, 0.316, 0.4, 0.5]

print(f"\nTesting reduction with 'girard' type:")
print(f"{'diagpercent':<15} {'Reduced Gens':<15} {'dHerror':<15} {'gredIdx len':<15}")
print("-" * 60)

for diagpercent in diagpercent_values:
    try:
        Z_red, dHerror, gredIdx = priv_reduceAdaptive(Z, diagpercent, 'girard')
        G_red = Z_red.generators()
        num_gens = G_red.shape[1] if G_red is not None and G_red.size > 0 else 0
        print(f"{diagpercent:<15.3f} {num_gens:<15} {dHerror:<15.6e} {len(gredIdx):<15}")
    except Exception as e:
        print(f"{diagpercent:<15.3f} ERROR: {e}")

print("\n" + "=" * 80)
print("Now we need to run the same test in MATLAB and compare")
print("=" * 80)
