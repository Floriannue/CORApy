"""test_abortion_condition - Test abortion condition with very small time steps"""

import numpy as np

def _aux_checkForAbortion(tVec, currt, tFinal):
    """MATLAB-exact abortion check"""
    abortAnalysis = False
    remTime = tFinal - currt
    N = 10
    k = len(tVec)
    if k == 0:
        return False
    lastNsteps = np.sum(tVec[max(0, k - N):])
    if lastNsteps == 0:
        abortAnalysis = True
    else:
        ratio = remTime / lastNsteps
        if ratio > 1e9:
            abortAnalysis = True
    return abortAnalysis

# Simulate scenario: very small time steps
# After 867 steps, we're at t=1.8, remaining time = 6.2
# If last 10 steps sum to something very small, ratio will be huge

# Scenario 1: Very small time steps (like Python is experiencing)
tVec_small = np.array([1e-10] * 867)  # All steps are 1e-10
currt_small = np.sum(tVec_small)
tFinal = 8.0

print("Scenario 1: Very small time steps (1e-10 each)")
print(f"  Total steps: {len(tVec_small)}")
print(f"  Current time: {currt_small:.6e}")
print(f"  Remaining time: {tFinal - currt_small:.6f}")
lastNsteps = np.sum(tVec_small[-10:])
print(f"  Last 10 steps sum: {lastNsteps:.6e}")
if lastNsteps > 0:
    ratio = (tFinal - currt_small) / lastNsteps
    print(f"  Ratio: {ratio:.2e}")
    print(f"  Would abort: {ratio > 1e9}")
abort = _aux_checkForAbortion(tVec_small, currt_small, tFinal)
print(f"  Abort result: {abort}")

# Scenario 2: Normal time steps (like MATLAB)
# MATLAB completes with 237 steps, so average step is ~8/237 = 0.0338
tVec_normal = np.array([0.0338] * 237)
currt_normal = np.sum(tVec_normal)

print("\nScenario 2: Normal time steps (~0.0338 each, like MATLAB)")
print(f"  Total steps: {len(tVec_normal)}")
print(f"  Current time: {currt_normal:.6f}")
print(f"  Remaining time: {tFinal - currt_normal:.6f}")
lastNsteps = np.sum(tVec_normal[-10:])
print(f"  Last 10 steps sum: {lastNsteps:.6f}")
if lastNsteps > 0:
    ratio = (tFinal - currt_normal) / lastNsteps
    print(f"  Ratio: {ratio:.2e}")
    print(f"  Would abort: {ratio > 1e9}")
abort = _aux_checkForAbortion(tVec_normal, currt_normal, tFinal)
print(f"  Abort result: {abort}")

# Scenario 3: Python's actual situation (867 steps, t=1.8)
# If we have 867 steps and t=1.8, average step is 1.8/867 = 0.002076
# But if last steps are much smaller...
tVec_python = np.concatenate([np.array([0.01] * 200), np.array([1e-9] * 667)])
currt_python = np.sum(tVec_python)

print("\nScenario 3: Python's situation (mixed: normal then very small)")
print(f"  Total steps: {len(tVec_python)}")
print(f"  Current time: {currt_python:.6f}")
print(f"  Remaining time: {tFinal - currt_python:.6f}")
lastNsteps = np.sum(tVec_python[-10:])
print(f"  Last 10 steps sum: {lastNsteps:.6e}")
if lastNsteps > 0:
    ratio = (tFinal - currt_python) / lastNsteps
    print(f"  Ratio: {ratio:.2e}")
    print(f"  Would abort: {ratio > 1e9}")
abort = _aux_checkForAbortion(tVec_python, currt_python, tFinal)
print(f"  Abort result: {abort}")
