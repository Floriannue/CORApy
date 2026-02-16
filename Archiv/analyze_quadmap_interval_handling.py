"""analyze_quadmap_interval_handling - Analyze how quadMap handles Interval matrices"""

import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval

print("=" * 80)
print("ANALYZING quadMap INTERVAL HANDLING")
print("=" * 80)

# Test case: Simple zonotope and Interval Hessian
print("\n1. Testing quadMap with Interval Hessian:")

# Create a simple zonotope
Z = Zonotope(np.array([[1.0], [0.0]]), np.array([[1.0, 0.5], [0.0, 0.3]]))

# Create an Interval Hessian (2x2 interval matrix)
# This simulates what happens when H[i] is an Interval
H_interval = Interval(
    np.array([[-0.1, -0.05], [-0.05, -0.1]]),  # inf
    np.array([[0.1, 0.05], [0.05, 0.1]])       # sup
)

H = [H_interval, H_interval]  # Two dimensions

print(f"Z center: {Z.center().flatten()}")
print(f"Z generators shape: {Z.generators().shape}")
print(f"H[0] type: {type(H[0])}")
print(f"H[0] inf max: {np.max(np.abs(H[0].inf))}")
print(f"H[0] sup max: {np.max(np.abs(H[0].sup))}")
print(f"H[0] center max: {np.max(np.abs(H[0].center()))}")

# Test quadMap
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

# Test the matrix multiplication step manually
print("\n2. Testing matrix multiplication manually:")
Zmat = np.hstack([Z.c, Z.G])
print(f"Zmat shape: {Zmat.shape}")

# Test: Zmat.T @ H[0] @ Zmat
try:
    # This is what Python does
    quadMat_python = Zmat.T @ H[0] @ Zmat
    print(f"quadMat_python type: {type(quadMat_python)}")
    if isinstance(quadMat_python, Interval):
        print(f"quadMat_python is Interval")
        print(f"  Center max: {np.max(np.abs(quadMat_python.center()))}")
        print(f"  Inf max: {np.max(np.abs(quadMat_python.inf))}")
        print(f"  Sup max: {np.max(np.abs(quadMat_python.sup))}")
        # Python uses center() for computation
        quadMat_center = quadMat_python.center()
        print(f"  Using center() for computation")
        print(f"  Center shape: {quadMat_center.shape}")
        print(f"  Center[0,0]: {quadMat_center[0,0]}")
        print(f"  Center diagonal: {np.diag(quadMat_center[1:, 1:])}")
except Exception as e:
    print(f"ERROR in manual test: {e}")
    import traceback
    traceback.print_exc()

# Compare with what MATLAB might do
print("\n3. MATLAB comparison:")
print("MATLAB: quadMat = Zmat'*Q{i}*Zmat")
print("  - If Q{i} is an Interval, MATLAB uses interval arithmetic")
print("  - Result is an Interval matrix")
print("  - MATLAB then extracts diagonal elements directly from Interval")
print("\nPython: quadMat = Zmat.T @ Q_i @ Zmat")
print("  - If Q_i is an Interval, Python uses Interval @ operator")
print("  - Result is an Interval matrix")
print("  - Python then uses center() to convert to numeric")
print("  - THIS MIGHT BE THE ISSUE: center() is an approximation!")

print("\n" + "=" * 80)
print("POTENTIAL ISSUE IDENTIFIED")
print("=" * 80)
print("Python uses quadMat.center() which is the MIDPOINT of the interval.")
print("This is a conservative approximation but might not match MATLAB's")
print("exact interval arithmetic handling.")
print("=" * 80)
