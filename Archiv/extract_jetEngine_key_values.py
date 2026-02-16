"""extract_jetEngine_key_values - Extract and compare key values from Python run"""

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

print("Running Python to extract key values...")
R = sys.reach(params, options)

# Extract key values from options
print("\n" + "=" * 80)
print("EXTRACTED KEY VALUES")
print("=" * 80)

if '_debug_finitehorizon' in options:
    print("\nFirst 20 finitehorizon computations:")
    for entry in options['_debug_finitehorizon'][:20]:
        print(f"\nStep {entry['step']}:")
        print(f"  prev_finitehorizon: {entry['prev_finitehorizon']:.6e}")
        print(f"  prev_varphi: {entry['prev_varphi']:.6f}")
        print(f"  zetaphi: {entry['zetaphi']:.6f}")
        print(f"  computed_finitehorizon: {entry['computed_finitehorizon']:.6e}")
        print(f"  remTime: {entry['remTime']:.6f}")
        print(f"  capped_finitehorizon: {entry['capped_finitehorizon']:.6e}")
        if entry['computed_finitehorizon'] > entry['remTime']:
            print(f"  ⚠️  UNBOUNDED: computed > remTime (not capped!)")

if 'finitehorizon' in options:
    fh_values = options['finitehorizon']
    print(f"\nfinitehorizon array (first 20):")
    for i in range(min(20, len(fh_values))):
        print(f"  Step {i+1}: {fh_values[i]:.6e}")

if 'varphi' in options:
    varphi_values = options['varphi']
    print(f"\nvarphi array (first 20):")
    for i in range(min(20, len(varphi_values))):
        if varphi_values[i] > 0:
            print(f"  Step {i+1}: {varphi_values[i]:.6f}")

if 'stepsize' in options:
    stepsize_values = options['stepsize']
    print(f"\nstepsize array (first 20):")
    for i in range(min(20, len(stepsize_values))):
        print(f"  Step {i+1}: {stepsize_values[i]:.6e}")

# Save for comparison
import pickle
with open('jetEngine_python_key_values.pkl', 'wb') as f:
    pickle.dump({
        'finitehorizon': options.get('finitehorizon', []),
        'varphi': options.get('varphi', []),
        'stepsize': options.get('stepsize', []),
        'debug_finitehorizon': options.get('_debug_finitehorizon', []),
    }, f)

print("\nKey values saved to jetEngine_python_key_values.pkl")
