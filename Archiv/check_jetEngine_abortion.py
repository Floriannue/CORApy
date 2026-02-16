"""check_jetEngine_abortion - Check abortion condition for jetEngine"""

import numpy as np
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
options['progress'] = False
options['progressInterval'] = 5
options['verbose'] = 0

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print("Running jetEngine...")
R = sys.reach(params, options)

# Extract results
if hasattr(R, 'timePoint') and R.timePoint is not None:
    if hasattr(R.timePoint, 'set'):
        timePoint_sets = R.timePoint.set
        numSteps = len(timePoint_sets)
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

# Get time steps from options (they should be stored there)
# Or compute from timePoint.time differences
if hasattr(R, 'tVec'):
    tVec = R.tVec
elif final_time is not None and numSteps > 0:
    # Compute from time differences
    times = R.timePoint.time if isinstance(R.timePoint.time, list) else [R.timePoint.time]
    tVec = np.diff([params['tStart']] + times)
else:
    tVec = []

print(f"\nResults:")
print(f"  Number of steps: {numSteps}")
print(f"  Final time: {final_time:.6f}")
print(f"  Expected: 8.0")
print(f"  Remaining time: {8.0 - final_time:.6f}")

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
    
    # Calculate abortion condition
    N = 10
    k = len(tVec)
    lastNsteps = np.sum(tVec[max(0, k - N):])
    remTime = 8.0 - final_time if final_time else 8.0
    
    print(f"\nAbortion condition check:")
    print(f"  Last {N} steps sum (lastNsteps) = {lastNsteps:.6e}")
    print(f"  Remaining time (remTime) = {remTime:.6f}")
    if lastNsteps > 0:
        ratio = remTime / lastNsteps
        print(f"  Ratio (remTime / lastNsteps) = {ratio:.2e}")
        print(f"  Abortion threshold: 1e9")
        print(f"  Would abort: {ratio > 1e9}")
        if ratio > 1e9:
            print(f"\n  ✓ Abortion condition is TRUE - this explains why Python stopped early!")
    else:
        print(f"  lastNsteps is zero - would abort immediately")
