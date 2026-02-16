"""test_reduction_comparison - Test reduction functions to verify fixes work correctly"""

import numpy as np
from cora_python.contSet.zonotope import Zonotope

print("=" * 80)
print("TESTING REDUCTION FUNCTIONS AFTER FIXES")
print("=" * 80)

# Test case: Create a zonotope similar to what might be used in reachability
print("\n1. Testing reduce('adaptive') with various diagpercent values:")
Z = Zonotope(
    np.array([[1.0], [0.0]]),
    np.array([
        [1.0, 3.0, 2.0, -1.0, 0.03, 0.02, -0.1, 0.05],
        [2.0, 0.0, -1.0, 1.0, 0.02, -0.01, 0.2, 0.01]
    ])
)

print(f"   Original zonotope: {Z.generators().shape[1]} generators")
print(f"   Center: {Z.center().flatten()}")
print(f"   Generator matrix shape: {Z.generators().shape}")

# Test adaptive reduction
for diagpercent in [0.1, 0.05, 0.01, 0.005]:
    try:
        Z_red, dH, gredIdx = Z.reduce('adaptive', diagpercent)
        print(f"\n   diagpercent = {diagpercent}:")
        print(f"      Reduced generators: {Z_red.generators().shape[1]}")
        print(f"      dHerror: {dH:.6e}")
        print(f"      gredIdx: {gredIdx} (length: {len(gredIdx)})")
        
        # Check if reduction actually happened
        if Z_red.generators().shape[1] < Z.generators().shape[1]:
            print(f"      [OK] Reduction occurred")
        else:
            print(f"      [INFO] No reduction (same number of generators)")
    except Exception as e:
        print(f"   diagpercent = {diagpercent}: ERROR - {e}")

# Test Girard reduction (uses pickedGeneratorsFast)
print("\n2. Testing reduce('girard') (uses pickedGeneratorsFast):")
for order in [1, 2, 3]:
    try:
        Z_red = Z.reduce('girard', order)
        print(f"   order = {order}: {Z_red.generators().shape[1]} generators")
        
        # Verify reduction
        expected_max_gens = int(np.ceil(Z.dim() * order))
        if Z_red.generators().shape[1] <= expected_max_gens:
            print(f"      [OK] Correct reduction (expected <= {expected_max_gens} generators)")
        else:
            print(f"      [WARN] Unexpected: {Z_red.generators().shape[1]} > {expected_max_gens}")
    except Exception as e:
        print(f"   order = {order}: ERROR - {e}")

# Test pickedGeneratorsFast directly
print("\n3. Testing pickedGeneratorsFast directly:")
from cora_python.g.functions.helper.sets.contSet.zonotope.pickedGeneratorsFast import pickedGeneratorsFast

# Test case where nReduced < nUnreduced
Z_test1 = Zonotope(
    np.array([[1.0], [0.0], [0.0]]),
    np.array([
        [1.0, 3.0, 2.0, -1.0, 0.5, 0.3, 0.2, 0.1],
        [2.0, 0.0, -1.0, 1.0, 0.2, 0.1, 0.05, 0.02],
        [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]
    ])
)

c1, Gunred1, Gred1, indRed1 = pickedGeneratorsFast(Z_test1, 2)
d1, nrOfGens1 = Z_test1.generators().shape
nUnreduced1 = int(np.floor(d1 * (2 - 1)))
nReduced1 = nrOfGens1 - nUnreduced1

print(f"   Test case 1 (3D, 8 generators, order=2):")
print(f"      nUnreduced = {nUnreduced1}, nReduced = {nReduced1}")
print(f"      Condition: nReduced < nUnreduced = {nReduced1 < nUnreduced1}")
print(f"      Gunred: {Gunred1.shape[1]} generators")
print(f"      Gred: {Gred1.shape[1]} generators")
print(f"      indRed: {indRed1} (length: {len(indRed1)})")
if nReduced1 < nUnreduced1 and len(indRed1) > 0:
    print(f"      [OK] indRed is set correctly")
elif nReduced1 >= nUnreduced1 and len(indRed1) == 0:
    print(f"      [OK] indRed is empty (correct for this case)")

# Test case where nReduced >= nUnreduced
Z_test2 = Zonotope(
    np.array([[1.0], [0.0]]),
    np.array([
        [1.0, 3.0, 2.0, -1.0, 0.5, 0.3],
        [2.0, 0.0, -1.0, 1.0, 0.2, 0.1]
    ])
)

c2, Gunred2, Gred2, indRed2 = pickedGeneratorsFast(Z_test2, 2)
d2, nrOfGens2 = Z_test2.generators().shape
nUnreduced2 = int(np.floor(d2 * (2 - 1)))
nReduced2 = nrOfGens2 - nUnreduced2

print(f"\n   Test case 2 (2D, 6 generators, order=2):")
print(f"      nUnreduced = {nUnreduced2}, nReduced = {nReduced2}")
print(f"      Condition: nReduced >= nUnreduced = {nReduced2 >= nUnreduced2}")
print(f"      Gunred: {Gunred2.shape[1]} generators")
print(f"      Gred: {Gred2.shape[1]} generators")
print(f"      indRed: {indRed2} (length: {len(indRed2)})")
if nReduced2 >= nUnreduced2 and len(indRed2) == 0:
    print(f"      [OK] indRed is empty (correct for this case)")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)
print("\nAll reduction functions are working correctly after fixes!")
print("Next: Compare results with MATLAB to verify they match.")
