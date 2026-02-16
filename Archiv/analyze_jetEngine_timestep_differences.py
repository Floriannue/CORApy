"""analyze_jetEngine_timestep_differences - Analyze why Python's time steps become so small"""

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

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print("Running Python to analyze time step progression...")
R = sys.reach(params, options)

if hasattr(R, 'tVec') and len(R.tVec) > 0:
    tVec = R.tVec
    
    print(f"\nTime step analysis:")
    print(f"  Total steps: {len(tVec)}")
    print(f"  Min: {np.min(tVec):.6e}")
    print(f"  Max: {np.max(tVec):.6e}")
    print(f"  Mean: {np.mean(tVec):.6e}")
    
    # Show first 20 steps
    print(f"\nFirst 20 time steps:")
    for i in range(min(20, len(tVec))):
        print(f"  Step {i+1}: dt={tVec[i]:.6e}")
    
    # Show steps around where they start getting very small
    print(f"\nTime steps around step 200 (where MATLAB would be ~halfway):")
    for i in range(max(0, 190), min(210, len(tVec))):
        print(f"  Step {i+1}: dt={tVec[i]:.6e}")
    
    # Show last 20 steps
    print(f"\nLast 20 time steps:")
    for i in range(max(0, len(tVec) - 20), len(tVec)):
        print(f"  Step {i+1}: dt={tVec[i]:.6e}")
    
    # Find when time steps become very small (< 1e-6)
    small_steps_idx = np.where(tVec < 1e-6)[0]
    if len(small_steps_idx) > 0:
        first_small = small_steps_idx[0]
        print(f"\nFirst very small time step (< 1e-6) at step {first_small + 1}")
        print(f"  dt = {tVec[first_small]:.6e}")
        if first_small > 0:
            print(f"  Previous dt = {tVec[first_small - 1]:.6e}")
            print(f"  Ratio = {tVec[first_small] / tVec[first_small - 1]:.6e}")
