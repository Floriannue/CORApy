"""Find matching reduction scenarios between Python and MATLAB"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("FINDING MATCHING REDUCTION SCENARIOS")
print("=" * 80)

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
py_reductions = []
for entry in python_upstream:
    if isinstance(entry, dict) and 'initReach_tracking' in entry:
        it = entry['initReach_tracking']
        if isinstance(it, dict):
            rhom_tp = it.get('Rhom_tp_num_generators')
            rend_tp = it.get('Rend_tp_num_generators')
            redFactor = it.get('redFactor')
            if rhom_tp and rend_tp:
                py_reductions.append({
                    'step': entry.get('step'),
                    'run': entry.get('run'),
                    'rhom_tp': rhom_tp,
                    'rend_tp': rend_tp,
                    'redFactor': redFactor,
                    'reduced': rhom_tp - rend_tp
                })

print(f"\nPython: Found {len(py_reductions)} entries with initReach_tracking")
print("First 10:")
for r in py_reductions[:10]:
    print(f"  Step {r['step']}, Run {r['run']}: {r['rhom_tp']} -> {r['rend_tp']} (reduced {r['reduced']}, redFactor={r['redFactor']})")

# MATLAB
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

mat_reductions = []
for entry in matlab_log:
    if hasattr(entry, 'initReach_tracking'):
        it = entry.initReach_tracking
        if isinstance(it, np.ndarray) and it.size > 0:
            it_obj = it[0]
            rhom_tp = getattr(it_obj, 'Rhom_tp_num_generators', None)
            rend_tp = getattr(it_obj, 'Rend_tp_num_generators', None)
            redFactor = getattr(it_obj, 'redFactor', None)
            if rhom_tp and rend_tp:
                mat_reductions.append({
                    'step': entry.step,
                    'run': entry.run,
                    'rhom_tp': rhom_tp,
                    'rend_tp': rend_tp,
                    'redFactor': redFactor,
                    'reduced': rhom_tp - rend_tp
                })

print(f"\nMATLAB: Found {len(mat_reductions)} entries with initReach_tracking")
print("First 10:")
for r in mat_reductions[:10]:
    print(f"  Step {r['step']}, Run {r['run']}: {r['rhom_tp']} -> {r['rend_tp']} (reduced {r['reduced']}, redFactor={r['redFactor']})")

# Find matching scenarios (same input, same redFactor)
print("\n" + "=" * 80)
print("FINDING MATCHING SCENARIOS")
print("=" * 80)

# Look for 5 generators input with redFactor 0.0005
target_rhom_tp = 5
target_redFactor = 0.0005

py_matches = [r for r in py_reductions if r['rhom_tp'] == target_rhom_tp and abs(r['redFactor'] - target_redFactor) < 1e-10]
mat_matches = [r for r in mat_reductions if r['rhom_tp'] == target_rhom_tp and abs(r['redFactor'] - target_redFactor) < 1e-10]

print(f"\nPython entries with {target_rhom_tp} generators, redFactor={target_redFactor}:")
for r in py_matches:
    print(f"  Step {r['step']}, Run {r['run']}: {r['rhom_tp']} -> {r['rend_tp']} (reduced {r['reduced']})")

print(f"\nMATLAB entries with {target_rhom_tp} generators, redFactor={target_redFactor}:")
for r in mat_matches:
    print(f"  Step {r['step']}, Run {r['run']}: {r['rhom_tp']} -> {r['rend_tp']} (reduced {r['reduced']})")

if py_matches and mat_matches:
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    py_avg_reduced = np.mean([r['reduced'] for r in py_matches])
    mat_avg_reduced = np.mean([r['reduced'] for r in mat_matches])
    print(f"Python average reduced: {py_avg_reduced:.1f} generators")
    print(f"MATLAB average reduced: {mat_avg_reduced:.1f} generators")
    print(f"Difference: {abs(py_avg_reduced - mat_avg_reduced):.1f} generators")
