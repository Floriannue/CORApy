"""compare_jetEngine_step_by_step - Track intermediate values in Python"""

import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine

dim_x = 2
params = {}
params['tFinal'] = 8.0
params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(dim_x))
params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
params['tStart'] = 0.0

options = {}
options['alg'] = 'lin-adaptive'
options['traceIntermediateValues'] = True  # Enable tracking

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print('Running Python with intermediate value tracking...')
R = sys.reach(params, options)

print('\nPython Results:')
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

print(f'  Number of steps: {numSteps}')
if final_time is not None:
    print(f'  Final time: {final_time:.10f}')

# Extract time steps
if hasattr(R, 'tVec'):
    tVec = R.tVec
    print(f'  Time step stats:')
    print(f'    Min: {np.min(tVec):.6e}')
    print(f'    Max: {np.max(tVec):.6e}')
    print(f'    Mean: {np.mean(tVec):.6e}')
    if len(tVec) >= 10:
        print(f'    Last 10 sum: {np.sum(tVec[-10:]):.6e}')
    
    # Check abortion condition at final step
    N = 10
    k = len(tVec)
    lastNsteps = np.sum(tVec[max(0, k - N):])
    remTime = params['tFinal'] - final_time if final_time else params['tFinal']
    if lastNsteps > 0:
        ratio = remTime / lastNsteps
        print(f'  Abortion check at final step:')
        print(f'    remTime: {remTime:.6f}')
        print(f'    lastNsteps: {lastNsteps:.6e}')
        print(f'    ratio: {ratio:.2e}')
        print(f'    Would abort: {ratio > 1e9}')
