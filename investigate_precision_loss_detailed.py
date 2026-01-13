"""
Detailed investigation of precision loss in initReach computation
Compares each step with MATLAB to identify where differences accumulate
"""
import numpy as np
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
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

print("=" * 80)
print("DETAILED PRECISION LOSS INVESTIGATION")
print("=" * 80)

# Step 1: Initial set
print("\n[STEP 1] Initial Set R0")
print(f"  R0 center: {params['R0'].c.flatten()}")
print(f"  R0 generators shape: {params['R0'].G.shape}")
print(f"  R0 first generator: {params['R0'].G[:, 0] if params['R0'].G.size > 0 else 'N/A'}")

# Step 2: Linearization
print("\n[STEP 2] Linearization")
sys, linsys, linParams, linOptions = linearize(tank, params['R0'], params, options)

print(f"  Linearization point p.x: {sys.linError.p.x.flatten()}")
print(f"  Linearization point p.u: {sys.linError.p.u.flatten()}")
print(f"  f0 (constant input): {sys.linError.f0.flatten()}")
print(f"  Jacobian A shape: {linsys.A.shape}")
print(f"  Jacobian A[0, :]: {linsys.A[0, :]}")
print(f"  Jacobian B shape: {linsys.B.shape if hasattr(linsys, 'B') else 'N/A'}")

# Step 3: Translate Rinit
print("\n[STEP 3] Translate Rinit")
Rdelta = params['R0'] + (-sys.linError.p.x)
print(f"  Rdelta center: {Rdelta.c.flatten()}")
print(f"  Rdelta generators shape: {Rdelta.G.shape}")
print(f"  Rdelta center precision check: {Rdelta.c.flatten()[:3]}")

# Step 4: Compute eAt
print("\n[STEP 4] Matrix Exponential eAt")
A = linsys.A
timeStep = options['timeStep']
from scipy.linalg import expm
eAt = expm(A * timeStep)
print(f"  eAt shape: {eAt.shape}")
print(f"  eAt[0, 0]: {eAt[0, 0]:.15e}")
print(f"  eAt[0, 1]: {eAt[0, 1]:.15e}")
print(f"  eAt max value: {np.max(eAt):.15e}")
print(f"  eAt min value: {np.min(eAt):.15e}")

# Step 5: Homogeneous solution
print("\n[STEP 5] Homogeneous Solution")
Htp, Hti, C_state = linsys.homogeneousSolution(Rdelta, timeStep, options['taylorTerms'])
print(f"  Htp center: {Htp.c.flatten()}")
print(f"  Htp generators shape: {Htp.G.shape}")
print(f"  Htp center first 3: {Htp.c.flatten()[:3]}")

# Check eAt * Rdelta manually
print("\n[STEP 5a] Manual eAt * Rdelta check")
eAt_c = eAt @ Rdelta.c
eAt_G = eAt @ Rdelta.G
print(f"  eAt @ Rdelta.c: {eAt_c.flatten()[:3]}")
print(f"  eAt @ Rdelta.G shape: {eAt_G.shape}")
print(f"  eAt @ Rdelta.G first column: {eAt_G[:, 0] if eAt_G.size > 0 else 'N/A'}")
print(f"  Htp.c matches eAt @ Rdelta.c? {np.allclose(Htp.c, eAt_c, rtol=1e-14)}")
if not np.allclose(Htp.c, eAt_c, rtol=1e-14):
    diff = Htp.c - eAt_c
    print(f"  Difference: {diff.flatten()[:3]}")
    print(f"  Max abs diff: {np.max(np.abs(diff)):.15e}")

# Step 6: Particular solutions
print("\n[STEP 6] Particular Solutions")
Pu, C_input_const, _ = linsys.particularSolution_constant(linParams['uTrans'], timeStep, options['taylorTerms'])
PU = linsys.particularSolution_timeVarying(linParams['U'], timeStep, options['taylorTerms'])
print(f"  Pu center: {Pu.c.flatten() if hasattr(Pu, 'c') else 'N/A'}")
print(f"  PU center: {PU.c.flatten() if hasattr(PU, 'c') else 'N/A'}")
print(f"  PU generators shape: {PU.G.shape if hasattr(PU, 'G') else 'N/A'}")

# Step 7: Zonotope addition
print("\n[STEP 7] Zonotope Addition (Rtp = Htp + PU + Pu)")
print(f"  Before addition:")
print(f"    Htp center: {Htp.c.flatten()[:3]}")
print(f"    PU center: {PU.c.flatten()[:3] if hasattr(PU, 'c') else 'N/A'}")
print(f"    Pu center: {Pu.c.flatten()[:3] if hasattr(Pu, 'c') else 'N/A'}")

# Manual addition to check precision
Htp_c = Htp.c.flatten()
PU_c = PU.c.flatten() if hasattr(PU, 'c') else np.zeros(dim_x)
Pu_c = Pu.c.flatten() if hasattr(Pu, 'c') else np.zeros(dim_x)
manual_sum_c = Htp_c + PU_c + Pu_c
print(f"  Manual sum of centers: {manual_sum_c[:3]}")

Rtp = Htp + PU + Pu
print(f"  Rtp center (after addition, BEFORE translation back): {Rtp.c.flatten()[:3]}")
print(f"  Rtp generators shape: {Rtp.G.shape}")
print(f"  Rtp center matches manual sum? {np.allclose(Rtp.c.flatten(), manual_sum_c, rtol=1e-14)}")
if not np.allclose(Rtp.c.flatten(), manual_sum_c, rtol=1e-14):
    diff = Rtp.c.flatten() - manual_sum_c
    print(f"  Difference: {diff[:3]}")
    print(f"  Max abs diff: {np.max(np.abs(diff)):.15e}")

# Step 7b: Translation back (this happens in linReach)
print("\n[STEP 7b] Translation Back by Linearization Point")
print(f"  Linearization point p.x: {sys.linError.p.x.flatten()[:3]}")
Rtp_translated = Rtp + sys.linError.p.x
print(f"  Rtp center (AFTER translation back): {Rtp_translated.c.flatten()[:3]}")
print(f"  Expected from manual: {manual_sum_c[:3] + sys.linError.p.x.flatten()[:3]}")
print(f"  Matches? {np.allclose(Rtp_translated.c.flatten(), manual_sum_c + sys.linError.p.x.flatten(), rtol=1e-14)}")

# Step 8: Interval conversion (on translated Rtp)
print("\n[STEP 8] Interval Conversion (on translated Rtp)")
print(f"  Rtp_translated center: {Rtp_translated.c.flatten()[:3]}")
print(f"  Rtp_translated generators shape: {Rtp_translated.G.shape}")
print(f"  Rtp_translated first generator: {Rtp_translated.G[:, 0] if Rtp_translated.G.size > 0 else 'N/A'}")

# Manual interval computation
from cora_python.g.functions.helper.precision.kahan_sum import kahan_sum_abs
delta_kahan = kahan_sum_abs(Rtp_translated.G, axis=1)
delta_standard = np.sum(np.abs(Rtp_translated.G), axis=1)
print(f"  delta (Kahan): {delta_kahan[:3]}")
print(f"  delta (standard): {delta_standard[:3]}")
print(f"  Difference Kahan vs standard: {delta_kahan[:3] - delta_standard[:3]}")
print(f"  Max abs diff: {np.max(np.abs(delta_kahan - delta_standard)):.15e}")

IH_tp = Interval(Rtp_translated)
print(f"  IH_tp.inf: {IH_tp.inf.flatten()[:3]}")
print(f"  IH_tp.sup: {IH_tp.sup.flatten()[:3]}")

# Manual interval bounds
manual_inf = Rtp_translated.c.flatten() - delta_kahan
manual_sup = Rtp_translated.c.flatten() + delta_kahan
print(f"  Manual inf: {manual_inf[:3]}")
print(f"  Manual sup: {manual_sup[:3]}")
print(f"  IH_tp.inf matches manual? {np.allclose(IH_tp.inf.flatten(), manual_inf, rtol=1e-14)}")
print(f"  IH_tp.sup matches manual? {np.allclose(IH_tp.sup.flatten(), manual_sup, rtol=1e-14)}")

# Step 9: Compare with expected MATLAB values
print("\n[STEP 9] Comparison with MATLAB Expected Values")
IH_tp_true_inf = np.array([[1.8057949711597598], [3.6433030183959114], [3.7940260617482671], 
                          [1.9519553317477598], [9.3409949650858550], [4.0928655724716370]])
IH_tp_true_sup = np.array([[2.2288356782079028], [4.0572873081850807], [4.1960714210115002], 
                          [2.3451418924166987], [9.7630596270322201], [4.4862797486713282]])

diff_inf = IH_tp.inf.flatten()[:6] - IH_tp_true_inf.flatten()
diff_sup = IH_tp.sup.flatten()[:6] - IH_tp_true_sup.flatten()
print(f"  Difference inf: {diff_inf}")
print(f"  Difference sup: {diff_sup}")
print(f"  Max abs diff inf: {np.max(np.abs(diff_inf)):.15e}")
print(f"  Max abs diff sup: {np.max(np.abs(diff_sup)):.15e}")

# Step 10: Trace back to find source
print("\n[STEP 10] Tracing Back to Find Source of Differences")
print("  Expected Rtp center (from MATLAB):")
expected_Rtp_c = (IH_tp_true_inf.flatten() + IH_tp_true_sup.flatten()) / 2
print(f"    {expected_Rtp_c[:3]}")
print(f"  Actual Rtp center (translated):")
actual_Rtp_c = Rtp_translated.c.flatten()[:6]
print(f"    {actual_Rtp_c[:3]}")
diff_Rtp_c = actual_Rtp_c - expected_Rtp_c
print(f"  Difference in Rtp center: {diff_Rtp_c[:3]}")
print(f"  Max abs diff: {np.max(np.abs(diff_Rtp_c)):.15e}")

print("\n  Expected delta (from MATLAB):")
expected_delta = (IH_tp_true_sup.flatten() - IH_tp_true_inf.flatten()) / 2
print(f"    {expected_delta[:3]}")
print(f"  Actual delta (Kahan):")
actual_delta = delta_kahan[:6]
print(f"    {actual_delta[:3]}")
diff_delta = actual_delta - expected_delta
print(f"  Difference in delta: {diff_delta[:3]}")
print(f"  Max abs diff: {np.max(np.abs(diff_delta)):.15e}")

print("\n" + "=" * 80)
print("SUMMARY: Precision Loss Analysis")
print("=" * 80)
print(f"1. Rtp center difference: {np.max(np.abs(diff_Rtp_c)):.15e}")
print(f"2. Delta (interval width) difference: {np.max(np.abs(diff_delta)):.15e}")
print(f"3. Final interval difference: {np.max([np.max(np.abs(diff_inf)), np.max(np.abs(diff_sup))]):.15e}")
print("\nThe largest source of difference is likely in:")
if np.max(np.abs(diff_Rtp_c)) > np.max(np.abs(diff_delta)):
    print("  -> Rtp center computation (zonotope operations)")
else:
    print("  -> Delta computation (interval conversion)")
