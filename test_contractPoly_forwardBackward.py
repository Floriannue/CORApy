"""Test contractPoly with forwardBackward algorithm"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractPoly import contractPoly
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

# Equivalent formulation with a polynomial function
c = -4
G = np.array([[1, 1]])
GI = np.array([])
E = 2 * np.eye(2)

print("=== Testing contractPoly with forwardBackward ===")
print(f"Initial domain: {dom}")

# Test with contract (should work)
result_contract = contract(f, dom, 'forwardBackward', 2, 2)
print(f"\ncontract result: {result_contract}")

# Test with contractPoly (should match)
result_contractPoly = contractPoly(c, G, GI, E, dom, 'forwardBackward', 2, 2)
print(f"contractPoly result: {result_contractPoly}")

if result_contractPoly is None:
    print("ERROR: contractPoly returned None!")
elif not np.allclose(result_contract.inf, result_contractPoly.inf, atol=1e-10):
    print(f"ERROR: infimum mismatch!")
    print(f"  contract: {result_contract.inf}")
    print(f"  contractPoly: {result_contractPoly.inf}")
elif not np.allclose(result_contract.sup, result_contractPoly.sup, atol=1e-10):
    print(f"ERROR: supremum mismatch!")
    print(f"  contract: {result_contract.sup}")
    print(f"  contractPoly: {result_contractPoly.sup}")
else:
    print("Success: Results match!")




