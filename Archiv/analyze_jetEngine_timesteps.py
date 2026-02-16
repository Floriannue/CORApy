"""analyze_jetEngine_timesteps - Analyze time step sizes to understand early abortion"""

import numpy as np
import time
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine

# Same parameters as test
dim_x = 2
params = {}
params['tFinal'] = 8.0
params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(dim_x))
params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
params['tStart'] = 0.0

options = {}
options['alg'] = 'lin-adaptive'
options['progress'] = False  # Disable progress to reduce noise
options['progressInterval'] = 5
options['verbose'] = 0

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print("Running jetEngine with time step tracking...")
print("=" * 80)

# Monkey patch to track time steps
original_reach = sys.reach

def tracked_reach(params, options):
    # Track time steps during computation
    tVec_all = []
    
    # We need to intercept the reach_adaptive function
    from cora_python.contDynamics.nonlinearSys import reach_adaptive
    import cora_python.contDynamics.nonlinearSys.reach_adaptive as reach_mod
    
    original_check = reach_mod._aux_checkForAbortion
    
    def tracked_check(tVec, currt, tFinal):
        # Store time steps
        if len(tVec) > len(tVec_all):
            tVec_all.extend(tVec[len(tVec_all):])
        
        result = original_check(tVec, currt, tFinal)
        
        # Log when abortion condition is checked near the end
        if len(tVec) >= 850 or result:
            remTime = tFinal - currt
            N = 10
            k = len(tVec)
            if k > 0:
                lastNsteps = np.sum(tVec[max(0, k - N):])
                if lastNsteps > 0:
                    ratio = remTime / lastNsteps
                else:
                    ratio = np.inf
                
                print(f"Abortion check at step {k}: t={currt:.6f}, remTime={remTime:.6f}, "
                      f"lastNsteps={lastNsteps:.6e}, ratio={ratio:.2e}, abort={result}")
        
        return result
    
    reach_mod._aux_checkForAbortion = tracked_check
    
    try:
        result = original_reach(params, options)
        
        # Get final time steps from result
        if hasattr(result, 'tVec'):
            tVec_final = result.tVec
        else:
            # Try to get from options or compute from timePoint
            tVec_final = tVec_all if tVec_all else []
        
        return result
    finally:
        reach_mod._aux_checkForAbortion = original_check

sys.reach = tracked_reach

# Run
adapTime = time.time()
R = sys.reach(params, options)
tComp = time.time() - adapTime

print("\n" + "=" * 80)
print("Analysis:")
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

print(f"  Number of steps: {numSteps}")
print(f"  Final time: {final_time:.6f}")
print(f"  Expected: 8.0")
print(f"  Remaining time: {8.0 - final_time:.6f}")

# Check abortion condition manually
if hasattr(R, 'tVec') and len(R.tVec) > 0:
    tVec = R.tVec
    print(f"\nTime step analysis (last 20 steps):")
    for i in range(max(0, len(tVec) - 20), len(tVec)):
        print(f"  Step {i+1}: dt={tVec[i]:.6e}")
    
    # Calculate abortion condition
    N = 10
    k = len(tVec)
    lastNsteps = np.sum(tVec[max(0, k - N):])
    remTime = 8.0 - final_time if final_time else 8.0
    if lastNsteps > 0:
        ratio = remTime / lastNsteps
        print(f"\nAbortion condition at final step:")
        print(f"  lastNsteps (sum of last {N} steps) = {lastNsteps:.6e}")
        print(f"  remTime = {remTime:.6f}")
        print(f"  ratio = remTime / lastNsteps = {ratio:.2e}")
        print(f"  Would abort if ratio > 1e9: {ratio > 1e9}")
