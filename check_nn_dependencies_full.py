"""Check if all ContSet methods needed by nn.verify are implemented"""
import numpy as np

print("=" * 70)
print("Checking ContSet dependencies for nn.verify")
print("=" * 70)

# Check PolyZonotope methods
print("\n1. PolyZonotope methods:")
print("-" * 70)
from cora_python.contSet.polyZonotope import PolyZonotope
pZ = PolyZonotope([1], [[1]], [[0]], [[1]])

poly_methods = {
    'representsa_': 'Used in nnActivationLayer and nnLinearLayer',
    'interval': 'Used in compBoundsPolyZono',
    'restructure': 'Used in restructurePolyZono',
    'compact_': 'Used in representsa_',
    'zonotope': 'Used in representsa_ and restructure',
    'dim': 'Basic method',
    'isemptyobject': 'Basic method'
}

for method, desc in poly_methods.items():
    exists = hasattr(pZ, method) and callable(getattr(pZ, method))
    status = "OK" if exists else "MISSING"
    print(f"{status:15} {method:20} - {desc}")

# Check Interval methods
print("\n2. Interval methods:")
print("-" * 70)
from cora_python.contSet.interval import Interval
I = Interval([1], [2])

interval_methods = {
    'interval': 'Used in compBoundsPolyZono (returns self)',
    'representsa_': 'Used in nnActivationLayer',
    'center': 'Used in conversionStarSetConZono, nnLinearLayer',
    'rad': 'Used in conversionStarSetConZono, nnLinearLayer',
    'inf': 'Property - used to access lower bounds',
    'sup': 'Property - used to access upper bounds'
}

for method, desc in interval_methods.items():
    if method in ['inf', 'sup']:
        exists = hasattr(I, method)
    else:
        exists = hasattr(I, method) and callable(getattr(I, method))
    status = "OK" if exists else "MISSING"
    print(f"{status:15} {method:20} - {desc}")

# Check Zonotope methods
print("\n3. Zonotope methods:")
print("-" * 70)
from cora_python.contSet.zonotope import Zonotope
Z = Zonotope([1], [[1]])

zonotope_methods = {
    'interval': 'Used in conversionStarSetConZono',
    'center': 'Used in nnLinearLayer',
    'generators': 'Used in restructure',
    'reduce': 'Used in restructure'
}
# Note: Zonotope doesn't have 'rad' - nnLinearLayer._rad() is only used for Interval objects

for method, desc in zonotope_methods.items():
    exists = hasattr(Z, method) and callable(getattr(Z, method))
    status = "OK" if exists else "MISSING"
    print(f"{status:15} {method:20} - {desc}")

# Check Polytope methods
print("\n4. Polytope methods:")
print("-" * 70)
from cora_python.contSet.polytope import Polytope
try:
    # Polytope constructor: Polytope(A, b) where A is (m x n) and b is (m x 1)
    P = Polytope(np.array([[1, 0], [0, 1]]), np.array([1, 1]))
    polytope_methods = {
        'interval': 'Used in conversionConZonoStarSet'
    }
    
    for method, desc in polytope_methods.items():
        exists = hasattr(P, method) and callable(getattr(P, method))
        status = "OK" if exists else "MISSING"
        print(f"{status:15} {method:20} - {desc}")
except Exception as e:
    print(f"ERROR: Could not create Polytope: {e}")

# Check ConZonotope methods
print("\n5. ConZonotope methods:")
print("-" * 70)
from cora_python.contSet.conZonotope import ConZonotope
try:
    cZ = ConZonotope([1], [[1]], [[1]], [1])
    conzonotope_methods = {
        'center': 'Used in conversionStarSetConZono',
        'G': 'Property - generator matrix',
        'A': 'Property - constraint matrix',
        'b': 'Property - constraint vector'
    }
    
    for method, desc in conzonotope_methods.items():
        if method in ['G', 'A', 'b']:
            exists = hasattr(cZ, method)
        else:
            exists = hasattr(cZ, method) and callable(getattr(cZ, method))
        status = "OK" if exists else "MISSING"
        print(f"{status:15} {method:20} - {desc}")
except Exception as e:
    print(f"ERROR: Could not create ConZonotope: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_ok = True
missing = []

# Check PolyZonotope
for method in poly_methods.keys():
    if not (hasattr(pZ, method) and callable(getattr(pZ, method))):
        all_ok = False
        missing.append(f"PolyZonotope.{method}")

# Check Interval
for method in interval_methods.keys():
    if method in ['inf', 'sup']:
        if not hasattr(I, method):
            all_ok = False
            missing.append(f"Interval.{method}")
    else:
        if not (hasattr(I, method) and callable(getattr(I, method))):
            all_ok = False
            missing.append(f"Interval.{method}")

# Check Zonotope
for method in zonotope_methods.keys():
    if not (hasattr(Z, method) and callable(getattr(Z, method))):
        all_ok = False
        missing.append(f"Zonotope.{method}")

if all_ok:
    print("\nOK: All required methods are available!")
else:
    print(f"\nMISSING METHODS: {', '.join(missing)}")
    print("These need to be implemented for nn.verify to work!")

print("=" * 70)

