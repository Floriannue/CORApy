"""track_jetEngine_key_values - Track varphi, finitehorizon, and timeStep values"""

import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine

# Monkey patch linReach_adaptive to track key values
import cora_python.contDynamics.nonlinearSys.linReach_adaptive as linReach_mod

original_linReach = linReach_mod.linReach_adaptive
tracked_data = []

def tracked_linReach(nlnsys, Rstart, params, options):
    step = options.get('i', 0)
    
    # Track key values before computation
    if step > 1:
        prev_finitehorizon = options.get('finitehorizon', [0])[step - 2] if 'finitehorizon' in options and len(options.get('finitehorizon', [])) >= step - 1 else 0
        prev_varphi = options.get('varphi', [0])[step - 2] if 'varphi' in options and len(options.get('varphi', [])) >= step - 1 else 0
        prev_timeStep = options.get('stepsize', [0])[step - 2] if 'stepsize' in options and len(options.get('stepsize', [])) >= step - 1 else 0
        minorder = int(options.get('minorder', 0))
        zetaphi_val = options.get('zetaphi', [0])[minorder] if 'zetaphi' in options and len(options.get('zetaphi', [])) > minorder else 0
        
        # Compute what finitehorizon should be
        if step > 1 and prev_finitehorizon > 0:
            computed_finitehorizon = prev_finitehorizon * (1 + prev_varphi - zetaphi_val)
            # Check if it would be capped
            remTime = params['tFinal'] - options.get('t', 0)
            capped_finitehorizon = min(remTime, computed_finitehorizon)
            
            tracked_data.append({
                'step': step,
                't': options.get('t', 0),
                'prev_finitehorizon': prev_finitehorizon,
                'prev_varphi': prev_varphi,
                'zetaphi': zetaphi_val,
                'computed_finitehorizon': computed_finitehorizon,
                'remTime': remTime,
                'capped_finitehorizon': capped_finitehorizon,
                'prev_timeStep': prev_timeStep,
            })
    
    # Call original
    result = original_linReach(nlnsys, Rstart, params, options)
    
    # Track after computation
    if step <= 20 or step % 50 == 0:
        actual_timeStep = options.get('timeStep', 0)
        actual_finitehorizon = options.get('finitehorizon', [0])[-1] if 'finitehorizon' in options and len(options.get('finitehorizon', [])) > 0 else 0
        
        if len(tracked_data) > 0 and tracked_data[-1]['step'] == step:
            tracked_data[-1]['actual_timeStep'] = actual_timeStep
            tracked_data[-1]['actual_finitehorizon'] = actual_finitehorizon
    
    return result

linReach_mod.linReach_adaptive = tracked_linReach

# Run
dim_x = 2
params = {}
params['tFinal'] = 8.0
params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(dim_x))
params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
params['tStart'] = 0.0

options = {}
options['alg'] = 'lin-adaptive'

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print("Tracking key values...")
R = sys.reach(params, options)

# Restore
linReach_mod.linReach_adaptive = original_linReach

# Analyze
print("\n" + "=" * 80)
print("KEY VALUES ANALYSIS")
print("=" * 80)

print("\nFirst 20 steps:")
for entry in tracked_data[:20]:
    print(f"\nStep {entry['step']}: t={entry['t']:.6f}")
    print(f"  prev_finitehorizon: {entry['prev_finitehorizon']:.6e}")
    print(f"  prev_varphi: {entry['prev_varphi']:.6f}")
    print(f"  zetaphi: {entry['zetaphi']:.6f}")
    print(f"  computed_finitehorizon: {entry['computed_finitehorizon']:.6e}")
    print(f"  remTime: {entry['remTime']:.6f}")
    print(f"  capped_finitehorizon: {entry['capped_finitehorizon']:.6e}")
    print(f"  prev_timeStep: {entry['prev_timeStep']:.6e}")
    if 'actual_timeStep' in entry:
        print(f"  actual_timeStep: {entry['actual_timeStep']:.6e}")
        if entry['computed_finitehorizon'] != entry['actual_timeStep']:
            print(f"  ⚠️  MISMATCH: computed={entry['computed_finitehorizon']:.6e}, actual={entry['actual_timeStep']:.6e}")

# Find when finitehorizon starts growing unbounded
print("\n" + "=" * 80)
print("FINITEHORIZON GROWTH ANALYSIS")
print("=" * 80)

for entry in tracked_data:
    if entry['computed_finitehorizon'] > entry['remTime'] * 10:
        print(f"\nStep {entry['step']}: finitehorizon ({entry['computed_finitehorizon']:.6e}) >> remTime ({entry['remTime']:.6f})")
        print(f"  This is when finitehorizon becomes unbounded (not capped)")
        break

# Check varphi values
print("\n" + "=" * 80)
print("VARPHI ANALYSIS")
print("=" * 80)

varphi_values = [e['prev_varphi'] for e in tracked_data if 'prev_varphi' in e and e['prev_varphi'] > 0]
if varphi_values:
    print(f"  First 10 varphi values: {varphi_values[:10]}")
    print(f"  Min varphi: {min(varphi_values):.6f}")
    print(f"  Max varphi: {max(varphi_values):.6f}")
    print(f"  Mean varphi: {np.mean(varphi_values):.6f}")
