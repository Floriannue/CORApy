"""track_optimaldeltat_python - Track _aux_optimaldeltat inputs and outputs in Python"""

import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine
import pickle

dim_x = 2
params = {}
params['tFinal'] = 8.0
params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(dim_x))
params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
params['tStart'] = 0.0

options = {}
options['alg'] = 'lin-adaptive'
options['trackOptimaldeltat'] = True  # Enable tracking

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print("Running Python with _aux_optimaldeltat tracking...")
R = sys.reach(params, options)

# Extract log
if 'optimaldeltatLog' in options:
    log = options['optimaldeltatLog']
    print(f"\nCaptured {len(log)} _aux_optimaldeltat calls")
    
    # Save to file
    with open('optimaldeltat_python_log.pkl', 'wb') as f:
        pickle.dump(log, f)
    print("Saved to optimaldeltat_python_log.pkl")
    
    # Show first 10 entries
    print("\nFirst 10 entries:")
    for i, entry in enumerate(log[:10]):
        print(f"\nStep {entry['step']}:")
        print(f"  deltat (finitehorizon): {entry['deltat']:.6e}")
        print(f"  varphimin: {entry['varphimin']:.6f}")
        print(f"  zetaP: {entry['zetaP']:.6f}")
        print(f"  rR: {entry['rR']:.6e}")
        print(f"  rerr1: {entry['rerr1']:.6e}")
        print(f"  varphiprod (first 5): {entry['varphiprod'][:5]}")
        print(f"  deltats (first 5): {entry['deltats'][:5]}")
        print(f"  objfuncset (first 5): {entry['objfuncset'][:5]}")
        print(f"  bestIdxnew: {entry['bestIdxnew']}")
        print(f"  deltatest (selected): {entry['deltatest']:.6e}")
        print(f"  kprimeest: {entry['kprimeest']:.6f}")
else:
    print("No _optimaldeltat_log found in options")
