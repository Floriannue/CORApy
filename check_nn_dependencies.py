"""Check if all PolyZonotope methods needed by nn.verify are implemented"""
from cora_python.contSet.polyZonotope import PolyZonotope
import numpy as np

pZ = PolyZonotope([1], [[1]], [[0]], [[1]])

methods_needed = {
    'representsa_': 'Used in nnActivationLayer and nnLinearLayer',
    'interval': 'Used in compBoundsPolyZono',
    'restructure': 'Used in restructurePolyZono',
    'compact_': 'Used in representsa_',
    'zonotope': 'Used in representsa_ and restructure',
    'dim': 'Basic method',
    'isemptyobject': 'Basic method'
}

print("Methods needed by nn.verify:")
print("=" * 60)
for method, description in methods_needed.items():
    exists = hasattr(pZ, method)
    status = "OK" if exists else "MISSING"
    print(f"{status:15} {method:20} - {description}")

print("\n" + "=" * 60)
missing = [m for m, desc in methods_needed.items() if not hasattr(pZ, m)]
if missing:
    print(f"\nMISSING METHODS: {', '.join(missing)}")
    print("These need to be implemented for nn.verify to work!")
else:
    print("\nAll required methods are available!")

