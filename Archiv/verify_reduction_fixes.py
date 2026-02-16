"""verify_reduction_fixes - Verify that reduce('adaptive') and pickedGeneratorsFast work correctly"""

import numpy as np
from cora_python.contSet.zonotope import Zonotope

print("=" * 80)
print("VERIFYING REDUCTION FIXES")
print("=" * 80)

# Test 1: reduce('adaptive')
print("\n1. Testing reduce('adaptive'):")
Z1 = Zonotope(np.array([[1], [0]]), np.array([[1, 3, 2, -1, 0.03, 0.02, -0.1], 
                                               [2, 0, -1, 1, 0.02, -0.01, 0.2]]))
print(f"   Original: {Z1.generators().shape[1]} generators")

for diagpercent in [0.1, 0.05, 0.01]:
    try:
        Z_red, dH, gredIdx = Z1.reduce('adaptive', diagpercent)
        print(f"   diagpercent={diagpercent}: {Z_red.generators().shape[1]} generators, "
              f"dH={dH:.6e}, gredIdx len={len(gredIdx)}")
    except Exception as e:
        print(f"   diagpercent={diagpercent}: ERROR - {e}")

# Test 2: reduce('girard') with pickedGeneratorsFast
print("\n2. Testing reduce('girard') (uses pickedGeneratorsFast):")
Z2 = Zonotope(np.array([[1], [0]]), np.array([[1, 3, 2, -1, 0.5, 0.3], 
                                               [2, 0, -1, 1, 0.2, 0.1]]))
print(f"   Original: {Z2.generators().shape[1]} generators")

for order in [1, 2, 3]:
    try:
        Z_red = Z2.reduce('girard', order)
        print(f"   order={order}: {Z_red.generators().shape[1]} generators")
    except Exception as e:
        print(f"   order={order}: ERROR - {e}")

# Test 3: Verify pickedGeneratorsFast handles different cases
print("\n3. Testing pickedGeneratorsFast cases:")
from cora_python.g.functions.helper.sets.contSet.zonotope.pickedGeneratorsFast import pickedGeneratorsFast

# Case: nReduced < nUnreduced
Z3 = Zonotope(np.array([[1], [0], [0]]), np.array([[1, 3, 2, -1, 0.5, 0.3, 0.2, 0.1], 
                                                     [2, 0, -1, 1, 0.2, 0.1, 0.05, 0.02],
                                                     [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]]))
c3, Gunred3, Gred3, indRed3 = pickedGeneratorsFast(Z3, 2)
d3, nrOfGens3 = Z3.generators().shape
nUnreduced3 = int(np.floor(d3 * (2 - 1)))
nReduced3 = nrOfGens3 - nUnreduced3
print(f"   Case 1 (nReduced < nUnreduced): nReduced={nReduced3}, nUnreduced={nUnreduced3}")
print(f"      Gunred: {Gunred3.shape[1]} generators, Gred: {Gred3.shape[1]} generators")
print(f"      indRed length: {len(indRed3)}")

# Case: nReduced >= nUnreduced
Z4 = Zonotope(np.array([[1], [0]]), np.array([[1, 3, 2, -1, 0.5, 0.3], 
                                               [2, 0, -1, 1, 0.2, 0.1]]))
c4, Gunred4, Gred4, indRed4 = pickedGeneratorsFast(Z4, 2)
d4, nrOfGens4 = Z4.generators().shape
nUnreduced4 = int(np.floor(d4 * (2 - 1)))
nReduced4 = nrOfGens4 - nUnreduced4
print(f"   Case 2 (nReduced >= nUnreduced): nReduced={nReduced4}, nUnreduced={nUnreduced4}")
print(f"      Gunred: {Gunred4.shape[1]} generators, Gred: {Gred4.shape[1]} generators")
print(f"      indRed length: {len(indRed4)} (should be empty)")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
