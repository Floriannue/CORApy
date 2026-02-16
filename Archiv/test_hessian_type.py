"""test_hessian_type - Test what type of Hessian is returned"""

import numpy as np
import sys
import os

# Add path
sys.path.insert(0, os.path.abspath('.'))

# Import jetEngine model
from cora_python.models.auxiliary.jetEngine import jetEngine
from cora_python.contDynamics.nonlinearSys import nonlinearSys

print("=" * 80)
print("TESTING HESSIAN TYPE")
print("=" * 80)

# Create system
sys_obj = nonlinearSys(jetEngine, 2, 1)

# Test 1: setHessian('standard')
print("\n1. Testing setHessian('standard'):")
sys_obj_std = sys_obj.setHessian('standard')
x = np.array([[1.0], [1.0]])
u = np.array([[0.0]])

try:
    H_std = sys_obj_std.hessian(x, u)
    print(f"  H type: {type(H_std)}")
    print(f"  H length: {len(H_std) if H_std else 0}")
    if H_std and len(H_std) > 0:
        print(f"  H[0] type: {type(H_std[0])}")
        print(f"  H[0] shape: {H_std[0].shape if hasattr(H_std[0], 'shape') else 'N/A'}")
        if hasattr(H_std[0], 'toarray'):
            print(f"  H[0] is sparse, converting to array...")
            H_std_0_dense = H_std[0].toarray()
            print(f"  H[0] (dense) max: {np.max(np.abs(H_std_0_dense))}")
        elif isinstance(H_std[0], np.ndarray):
            print(f"  H[0] (dense) max: {np.max(np.abs(H_std[0]))}")
        else:
            print(f"  H[0] is not array or sparse: {H_std[0]}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: setHessian('int')
print("\n2. Testing setHessian('int'):")
sys_obj_int = sys_obj.setHessian('int')
from cora_python.contSet.interval import Interval
x_int = Interval(np.array([[0.9], [0.9]]), np.array([[1.1], [1.1]]))
u_int = Interval(np.array([[-0.1]]), np.array([[0.1]]))

try:
    H_int = sys_obj_int.hessian(x_int, u_int)
    print(f"  H type: {type(H_int)}")
    print(f"  H length: {len(H_int) if H_int else 0}")
    if H_int and len(H_int) > 0:
        print(f"  H[0] type: {type(H_int[0])}")
        if hasattr(H_int[0], 'inf') and hasattr(H_int[0], 'sup'):
            print(f"  H[0] is Interval")
            print(f"  H[0] inf max: {np.max(np.abs(H_int[0].inf)) if hasattr(H_int[0].inf, '__abs__') else 'N/A'}")
            print(f"  H[0] sup max: {np.max(np.abs(H_int[0].sup)) if hasattr(H_int[0].sup, '__abs__') else 'N/A'}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("EXPECTED:")
print("  'standard' should return NUMERIC (sparse or dense matrix)")
print("  'int' should return INTERVAL")
print("=" * 80)
