"""Debug contractPoly splitting logic"""
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

print("=== Testing contractPoly with forwardBackward and splits=2 ===")
print(f"Initial domain: {dom}")

# Test with contract (should work)
print("\n--- contract with forwardBackward, splits=2 ---")
result_contract = contract(f, dom, 'forwardBackward', 2, 2)
print(f"Result: {result_contract}")

# Test with contractPoly (should match)
print("\n--- contractPoly with forwardBackward, splits=2 ---")
result_contractPoly = contractPoly(c, G, GI, E, dom, 'forwardBackward', 2, 2)
print(f"Result: {result_contractPoly}")

# Test with contractPoly without splitting (should work)
print("\n--- contractPoly with forwardBackward, splits=None ---")
result_contractPoly_no_split = contractPoly(c, G, GI, E, dom, 'forwardBackward', 2, None)
print(f"Result: {result_contractPoly_no_split}")

# Test with contractPoly with splits=1
print("\n--- contractPoly with forwardBackward, splits=1 ---")
result_contractPoly_split1 = contractPoly(c, G, GI, E, dom, 'forwardBackward', 2, 1)
print(f"Result: {result_contractPoly_split1}")







