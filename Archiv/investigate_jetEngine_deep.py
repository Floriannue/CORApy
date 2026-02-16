"""investigate_jetEngine_deep - Deep investigation of why Python's time steps become small"""

import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine

# Monkey patch to track key intermediate values
import cora_python.contDynamics.nonlinearSys.linReach_adaptive as linReach_mod

# Store original function
original_linReach = linReach_mod.linReach_adaptive

# Track key values
tracked_values = []

def tracked_linReach(nlnsys, Rstart, params, options):
    """Wrapper to track intermediate values"""
    step = options.get('i', 0)
    
    # Track before computation
    if step <= 10 or step % 50 == 0:
        tracked_values.append({
            'step': step,
            'before': {
                't': options.get('t', 0),
                'timeStep': options.get('timeStep', 0),
                'error_adm_horizon_max': np.max(options.get('error_adm_horizon', np.array([0]))) if 'error_adm_horizon' in options else 0,
            }
        })
    
    # Call original
    result = original_linReach(nlnsys, Rstart, params, options)
    
    # Track after computation
    if step <= 10 or step % 50 == 0:
        if len(tracked_values) > 0 and tracked_values[-1]['step'] == step:
            tracked_values[-1]['after'] = {
                'timeStep': options.get('timeStep', 0),
                'finitehorizon': options.get('finitehorizon', [0])[-1] if 'finitehorizon' in options and len(options.get('finitehorizon', [])) > 0 else 0,
                'varphi': options.get('varphi', [0])[-1] if 'varphi' in options and len(options.get('varphi', [])) > 0 else 0,
            }
    
    return result

# Replace function
linReach_mod.linReach_adaptive = tracked_linReach

# Run test
dim_x = 2
params = {}
params['tFinal'] = 8.0
params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(dim_x))
params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
params['tStart'] = 0.0

options = {}
options['alg'] = 'lin-adaptive'

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print("Running Python with deep tracking...")
R = sys.reach(params, options)

# Restore original
linReach_mod.linReach_adaptive = original_linReach

# Analyze tracked values
print("\n" + "=" * 80)
print("TRACKED INTERMEDIATE VALUES")
print("=" * 80)

for entry in tracked_values[:20]:  # Show first 20
    print(f"\nStep {entry['step']}:")
    if 'before' in entry:
        print(f"  Before: t={entry['before']['t']:.6f}, timeStep={entry['before']['timeStep']:.6e}, error_adm_max={entry['before']['error_adm_horizon_max']:.6e}")
    if 'after' in entry:
        print(f"  After: timeStep={entry['after']['timeStep']:.6e}, finitehorizon={entry['after']['finitehorizon']:.6e}, varphi={entry['after']['varphi']:.6e}")

# Get final results
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

print(f"\n" + "=" * 80)
print(f"Final Results: {numSteps} steps, final time: {final_time:.6f}")
