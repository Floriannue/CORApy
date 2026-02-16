"""debug_jetEngine_abortion - Debug why Python aborts early for jetEngine"""

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
options['progress'] = True
options['progressInterval'] = 5
options['verbose'] = 1

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print("Running jetEngine with abortion tracking...")
print("=" * 80)

# Track time steps and abortion condition
tVec_tracking = []
abortion_checks = []

# Monkey patch _aux_checkForAbortion to track calls
from cora_python.contDynamics.nonlinearSys.reach_adaptive import _aux_checkForAbortion as original_check

def tracked_check(tVec, currt, tFinal):
    result = original_check(tVec, currt, tFinal)
    
    # Calculate abortion condition details
    remTime = tFinal - currt
    N = 10
    k = len(tVec)
    if k > 0:
        lastNsteps = np.sum(tVec[max(0, k - N):])
        if lastNsteps > 0:
            ratio = remTime / lastNsteps
        else:
            ratio = np.inf
        
        abortion_checks.append({
            'step': k,
            'currt': currt,
            'remTime': remTime,
            'lastNsteps': lastNsteps,
            'ratio': ratio,
            'abort': result
        })
        
        if result or k % 50 == 0 or k >= 800:  # Log every 50 steps or near end
            print(f"Step {k}: t={currt:.6f}, remTime={remTime:.6f}, "
                  f"lastNsteps={lastNsteps:.6e}, ratio={ratio:.2e}, abort={result}")
    
    return result

# Replace the function temporarily
import cora_python.contDynamics.nonlinearSys.reach_adaptive as reach_mod
reach_mod._aux_checkForAbortion = tracked_check

# Run reachability
adapTime = time.time()
R = sys.reach(params, options)
tComp = time.time() - adapTime

print("\n" + "=" * 80)
print("Results:")
print(f"  Computation time: {tComp:.2f} seconds")
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

print(f"  Number of steps: {numSteps}")
if final_time is not None:
    print(f"  Final time: {final_time:.6f}")
    print(f"  Expected: 8.0")
    print(f"  Difference: {8.0 - final_time:.6f}")

print("\nAbortion condition analysis:")
if abortion_checks:
    # Show last 10 checks
    print("\nLast 10 abortion checks:")
    for check in abortion_checks[-10:]:
        print(f"  Step {check['step']}: t={check['currt']:.6f}, "
              f"remTime={check['remTime']:.6f}, lastNsteps={check['lastNsteps']:.6e}, "
              f"ratio={check['ratio']:.2e}, abort={check['abort']}")
    
    # Find when abortion was triggered
    aborted_checks = [c for c in abortion_checks if c['abort']]
    if aborted_checks:
        print(f"\nAbortion triggered at step {aborted_checks[0]['step']}")
        print(f"  t={aborted_checks[0]['currt']:.6f}")
        print(f"  remTime={aborted_checks[0]['remTime']:.6f}")
        print(f"  lastNsteps={aborted_checks[0]['lastNsteps']:.6e}")
        print(f"  ratio={aborted_checks[0]['ratio']:.2e}")
