"""test_jetEngine_python - Run jetEngine adaptive reachability in Python
This matches test_nonlinearSys_reach_adaptive_01_jetEngine.py and test_jetEngine_matlab.m"""

import numpy as np
import time
import pickle
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine

# system dimension
dim_x = 2

# parameters
params = {}
params['tFinal'] = 8.0
params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(dim_x))
params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
params['tStart'] = 0.0

# algorithm parameters
options = {}
options['alg'] = 'lin-adaptive'
options['progress'] = True
options['progressInterval'] = 5
options['verbose'] = 1

# init system
sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

# run reachability analysis
print('Starting Python reachability analysis...')
adapTime = time.time()
R = sys.reach(params, options)  # Returns ReachSet object
tComp = time.time() - adapTime

print(f'Python computation completed in {tComp:.2f} seconds')

# Extract results
if hasattr(R, 'timePoint') and R.timePoint is not None:
    if hasattr(R.timePoint, 'set'):
        timePoint_sets = R.timePoint.set
    elif isinstance(R.timePoint, dict):
        timePoint_sets = R.timePoint.get('set', [])
    else:
        timePoint_sets = []
    
    if len(timePoint_sets) > 0:
        endset = timePoint_sets[-1]
        gamma_o = 2 * endset.interval().rad()
        
        print(f'Final set radius: {np.max(gamma_o):.6e}')
        print(f'Number of time points: {len(timePoint_sets)}')
        
        if hasattr(R.timePoint, 'time'):
            final_time = R.timePoint.time[-1] if isinstance(R.timePoint.time, list) else R.timePoint.time
            print(f'Final time: {final_time:.6f}')
        else:
            final_time = None
    else:
        print('ERROR: No time point sets computed')
        gamma_o = None
        final_time = None
else:
    print('ERROR: No time point data available')
    timePoint_sets = []
    gamma_o = None
    final_time = None

# Get final options (may be modified during computation)
# Note: options dict is modified in-place, so we need to get it from R if available
final_alg = options.get('alg', 'unknown')  # Should be 'lin' after 'adaptive' removal

# Save key results for comparison
results = {
    'tComp': tComp,
    'numSteps': len(timePoint_sets),
    'finalTime': final_time,
    'finalRadius': float(np.max(gamma_o)) if gamma_o is not None else None,
    'options_alg': final_alg
}

with open('jetEngine_python_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'R': R, 'options': options}, f)

print('\nResults saved to jetEngine_python_results.pkl')
print('Key results:')
print(f'  Computation time: {results["tComp"]:.2f} seconds')
print(f'  Number of steps: {results["numSteps"]}')
if results['finalTime'] is not None:
    print(f'  Final time: {results["finalTime"]:.6f}')
if results['finalRadius'] is not None:
    print(f'  Final radius: {results["finalRadius"]:.6e}')
print(f'  Final alg: {results["options_alg"]}')
