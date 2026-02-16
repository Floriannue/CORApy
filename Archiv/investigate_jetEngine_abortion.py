"""investigate_jetEngine_abortion - Investigate why Python aborts early for jetEngine"""

import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine

# Exact same parameters as MATLAB
dim_x = 2
params = {}
params['tFinal'] = 8.0
params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(dim_x))
params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
params['tStart'] = 0.0

options = {}
options['alg'] = 'lin-adaptive'

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print("Running jetEngine to investigate abortion...")
print("=" * 80)

# Monkey patch to track abortion checks
from cora_python.contDynamics.nonlinearSys.reach_adaptive import reach_adaptive
import cora_python.contDynamics.nonlinearSys.reach_adaptive as reach_mod

original_check = reach_mod._aux_checkForAbortion
abortion_log = []

def tracked_check(tVec, currt, tFinal):
    result = original_check(tVec, currt, tFinal)
    
    remTime = tFinal - currt
    N = 10
    k = len(tVec)
    if k > 0:
        lastNsteps = np.sum(tVec[max(0, k - N):])
        if lastNsteps > 0:
            ratio = remTime / lastNsteps
        else:
            ratio = np.inf
        
        # Log when abortion is triggered or near trigger
        if result or (k >= 200 and (k % 50 == 0 or ratio > 1e8)):
            abortion_log.append({
                'step': k,
                'currt': currt,
                'remTime': remTime,
                'lastNsteps': lastNsteps,
                'ratio': ratio,
                'abort': result,
                'last_dt': tVec[-1] if len(tVec) > 0 else 0
            })
    
    return result

reach_mod._aux_checkForAbortion = tracked_check

# Run
R = sys.reach(params, options)

# Restore original
reach_mod._aux_checkForAbortion = original_check

# Analyze results
if hasattr(R, 'timePoint') and R.timePoint is not None:
    if hasattr(R.timePoint, 'set'):
        numSteps = len(R.timePoint.set)
        if hasattr(R.timePoint, 'time'):
            final_time = R.timePoint.time[-1] if isinstance(R.timePoint.time, list) else R.timePoint.time
        else:
            final_time = None
    else:
        numSteps = 0
        final_time = None
else:
    numSteps = 0
    final_time = None

print(f"\nResults:")
print(f"  Number of steps: {numSteps}")
print(f"  Final time: {final_time:.6f}")
print(f"  Expected: 8.0")
print(f"  Remaining: {8.0 - final_time:.6f}")

# Get time steps
if hasattr(R, 'tVec'):
    tVec = R.tVec
else:
    tVec = []

if len(tVec) > 0:
    print(f"\nTime step analysis:")
    print(f"  Total steps: {len(tVec)}")
    print(f"  Min time step: {np.min(tVec):.6e}")
    print(f"  Max time step: {np.max(tVec):.6e}")
    print(f"  Mean time step: {np.mean(tVec):.6e}")
    
    # Show last 20 time steps
    print(f"\nLast 20 time steps:")
    for i in range(max(0, len(tVec) - 20), len(tVec)):
        print(f"  Step {i+1}: dt={tVec[i]:.6e}")
    
    # Calculate abortion condition at final step
    N = 10
    k = len(tVec)
    lastNsteps = np.sum(tVec[max(0, k - N):])
    remTime = 8.0 - final_time if final_time else 8.0
    
    print(f"\nAbortion condition at final step:")
    print(f"  Last {N} steps sum (lastNsteps) = {lastNsteps:.6e}")
    print(f"  Remaining time (remTime) = {remTime:.6f}")
    if lastNsteps > 0:
        ratio = remTime / lastNsteps
        print(f"  Ratio (remTime / lastNsteps) = {ratio:.2e}")
        print(f"  Abortion threshold: 1e9")
        print(f"  Would abort: {ratio > 1e9}")
        if ratio > 1e9:
            print(f"\n  ✓ Abortion condition is TRUE - this explains early termination!")
    else:
        print(f"  lastNsteps is zero - would abort immediately")

print(f"\nAbortion check log (showing when abortion was triggered or near):")
if abortion_log:
    for entry in abortion_log[-10:]:  # Show last 10
        print(f"  Step {entry['step']}: t={entry['currt']:.6f}, "
              f"remTime={entry['remTime']:.6f}, lastNsteps={entry['lastNsteps']:.6e}, "
              f"ratio={entry['ratio']:.2e}, last_dt={entry['last_dt']:.6e}, abort={entry['abort']}")
else:
    print("  No abortion checks logged (abortion may have been triggered before logging started)")
