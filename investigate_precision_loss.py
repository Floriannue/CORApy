"""
Investigate where precision is lost in the computation chain
Compare each step between Python and MATLAB to find precision bottlenecks
"""
import numpy as np
from scipy.linalg import expm
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.zonotope.center import center
from cora_python.contDynamics.nonlinearSys.linearize import linearize
from cora_python.contDynamics.linearSys.oneStep import oneStep

# Setup
dim_x = 6
params = {
    'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
    'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1)),
    'tFinal': 4,
    'uTrans': np.zeros((1, 1))
}

options = {
    'timeStep': 4,
    'taylorTerms': 4,
    'zonotopeOrder': 50,
    'alg': 'lin',
    'tensorOrder': 2,
    'maxError': np.full((dim_x, 1), np.inf)
}

# System
tank = NonlinearSys(tank6Eq, states=6, inputs=1)

# Compute derivatives
from cora_python.contDynamics.contDynamics.derivatives import derivatives
derivatives(tank, options)

# Linearize
sys, linsys, linParams, linOptions = linearize(tank, params['R0'], params, options)

# Translate Rinit
Rdelta = params['R0'] + (-sys.linError.p.x)

print("=== PRECISION LOSS INVESTIGATION ===\n")

# Step 1: Check eAt precision (we know this is perfect)
A = linsys.A
timeStep = options['timeStep']
eAt = expm(A * timeStep)
print("Step 1: eAt computation")
print(f"  eAt[0, 0]: {eAt[0, 0]:.15e}")
print(f"  Precision: Machine precision (verified)")

# Step 2: Check Rdelta precision
print("\nStep 2: Rdelta (translated initial set)")
print(f"  Rdelta center: {Rdelta.c.flatten()}")
print(f"  Rdelta generators shape: {Rdelta.G.shape}")
print(f"  Rdelta center precision: Check if matches MATLAB exactly")

# Step 3: Check eAt * Rdelta (homogeneous solution)
print("\nStep 3: eAt * Rdelta (homogeneous solution)")
Htp, Hti, C_state = linsys.homogeneousSolution(Rdelta, timeStep, options['taylorTerms'])
print(f"  Htp center: {Htp.c.flatten()}")
print(f"  Htp generators shape: {Htp.G.shape}")
print(f"  Htp center precision: Check matrix multiplication precision")

# Step 4: Check particular solutions
print("\nStep 4: Particular solutions")
Pu, C_input_const, _ = linsys.particularSolution_constant(linParams['uTrans'], timeStep, options['taylorTerms'])
PU = linsys.particularSolution_timeVarying(linParams['U'], timeStep, options['taylorTerms'])
print(f"  Pu center: {Pu.c.flatten() if hasattr(Pu, 'c') else 'N/A'}")
print(f"  PU center: {PU.c.flatten() if hasattr(PU, 'c') else 'N/A'}")

# Step 5: Check final reachable set
print("\nStep 5: Final reachable set (Rtp = Htp + PU + Pu)")
Rtp = Htp + PU + Pu
print(f"  Rtp center: {Rtp.c.flatten()}")
print(f"  Rtp generators shape: {Rtp.G.shape}")
print(f"  Rtp center precision: Check zonotope addition precision")

# Step 6: Check interval conversion
print("\nStep 6: Interval conversion")
IH_tp = Interval(Rtp)
print(f"  IH_tp.inf: {IH_tp.inf.flatten()[:6]}")
print(f"  IH_tp.sup: {IH_tp.sup.flatten()[:6]}")
print(f"  Interval precision: Check sum(abs(G)) precision")

# Check for potential precision issues
print("\n=== PRECISION ISSUES TO INVESTIGATE ===")
print("1. Zonotope addition: Generator concatenation may accumulate errors")
print("2. Matrix-vector multiplication: eAt * Z may have rounding errors")
print("3. Interval conversion: sum(abs(G)) may accumulate rounding")
print("4. Generator reduction: May introduce small errors")

# Check if using higher precision would help
print("\n=== PRECISION IMPROVEMENT OPTIONS ===")
print("1. Use np.float128 for critical computations")
print("2. Use Kahan summation for generator sums")
print("3. Minimize intermediate rounding")
print("4. Use compensated arithmetic for additions")
