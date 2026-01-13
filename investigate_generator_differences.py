"""
Investigate generator differences between Python and MATLAB
"""
import numpy as np
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys.linearize import linearize
from cora_python.contDynamics.linearSys.oneStep import oneStep
from scipy.linalg import expm

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

# Compute eAt
A = linsys.A
timeStep = options['timeStep']
eAt = expm(A * timeStep)

# Compute homogeneous solution
Htp, Hti, C_state = linsys.homogeneousSolution(Rdelta, timeStep, options['taylorTerms'])

# Particular solutions
Pu, C_input_const, _ = linsys.particularSolution_constant(linParams['uTrans'], timeStep, options['taylorTerms'])
PU = linsys.particularSolution_timeVarying(linParams['U'], timeStep, options['taylorTerms'])

# Combine
Rtp = Htp + PU + Pu
Rtp_translated = Rtp + sys.linError.p.x

print("=" * 80)
print("GENERATOR DIFFERENCES INVESTIGATION")
print("=" * 80)

# Check eAt * Rdelta.G manually
print("\n[1] eAt * Rdelta.G (homogeneous generators)")
eAt_G = eAt @ Rdelta.G
print(f"  eAt shape: {eAt.shape}")
print(f"  Rdelta.G shape: {Rdelta.G.shape}")
print(f"  eAt @ Rdelta.G shape: {eAt_G.shape}")
print(f"  eAt @ Rdelta.G[:, 0]: {eAt_G[:, 0]}")
print(f"  Htp.G[:, 0]: {Htp.G[:, 0]}")
print(f"  Match? {np.allclose(eAt_G, Htp.G, rtol=1e-14)}")
if not np.allclose(eAt_G, Htp.G, rtol=1e-14):
    diff = Htp.G - eAt_G
    print(f"  Max abs diff: {np.max(np.abs(diff)):.15e}")
    print(f"  Diff[:, 0]: {diff[:, 0]}")

# Check PU generators
print("\n[2] PU generators (time-varying input)")
print(f"  PU.G shape: {PU.G.shape}")
print(f"  PU.G[:, 0]: {PU.G[:, 0] if PU.G.size > 0 else 'N/A'}")

# Check Pu generators
print("\n[3] Pu generators (constant input)")
if hasattr(Pu, 'G'):
    print(f"  Pu.G shape: {Pu.G.shape}")
    print(f"  Pu.G[:, 0]: {Pu.G[:, 0] if Pu.G.size > 0 else 'N/A'}")
else:
    print(f"  Pu is numeric: {Pu}")

# Check combined generators
print("\n[4] Combined Rtp generators")
print(f"  Rtp.G shape: {Rtp.G.shape}")
print(f"  Rtp.G[:, 0]: {Rtp.G[:, 0]}")
print(f"  Rtp.G[:, 6]: {Rtp.G[:, 6] if Rtp.G.shape[1] > 6 else 'N/A'}")

# Check generator sums
print("\n[5] Generator sums (delta computation)")
from cora_python.g.functions.helper.precision.kahan_sum import kahan_sum_abs
delta_kahan = kahan_sum_abs(Rtp_translated.G, axis=1)
delta_standard = np.sum(np.abs(Rtp_translated.G), axis=1)
print(f"  delta (Kahan): {delta_kahan[:3]}")
print(f"  delta (standard): {delta_standard[:3]}")
print(f"  Difference: {delta_kahan[:3] - delta_standard[:3]}")
print(f"  Max abs diff: {np.max(np.abs(delta_kahan - delta_standard)):.15e}")

# Expected delta from MATLAB
expected_delta = np.array([0.21152035, 0.20699214, 0.20102268, 0.19653053, 0.21105333, 0.19670709])
print(f"\n  Expected delta (MATLAB): {expected_delta[:3]}")
print(f"  Actual delta (Python): {delta_kahan[:3]}")
print(f"  Difference: {delta_kahan[:3] - expected_delta[:3]}")
print(f"  Max abs diff: {np.max(np.abs(delta_kahan[:6] - expected_delta)):.15e}")

# Check individual generator contributions
print("\n[6] Individual generator contributions")
abs_G = np.abs(Rtp_translated.G)
print(f"  abs(G) shape: {abs_G.shape}")
print(f"  abs(G)[0, :]: {abs_G[0, :]}")
print(f"  sum(abs(G)[0, :]): {np.sum(abs_G[0, :]):.15e}")
print(f"  delta[0]: {delta_kahan[0]:.15e}")
print(f"  Match? {np.allclose(np.sum(abs_G[0, :]), delta_kahan[0], rtol=1e-14)}")

# Check if generators are different due to matrix multiplication order
print("\n[7] Matrix multiplication precision check")
# Check if eAt * G has precision issues
test_G = Rdelta.G[:, 0:1]  # First generator
eAt_g = eAt @ test_G
print(f"  eAt @ G[:, 0] (first generator): {eAt_g.flatten()[:3]}")
print(f"  Htp.G[:, 0]: {Htp.G[:, 0][:3]}")
print(f"  Match? {np.allclose(eAt_g.flatten(), Htp.G[:, 0], rtol=1e-14)}")

# Check all generators
print("\n[8] All generators comparison")
for i in range(min(6, Rdelta.G.shape[1])):
    eAt_gi = eAt @ Rdelta.G[:, i:i+1]
    diff = eAt_gi.flatten() - Htp.G[:, i]
    max_diff = np.max(np.abs(diff))
    if max_diff > 1e-14:
        print(f"  Generator {i}: max diff = {max_diff:.15e}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("The precision loss is in the delta computation (sum of abs of generators).")
print("This suggests the generators themselves may have small differences,")
print("which accumulate when summing their absolute values.")
