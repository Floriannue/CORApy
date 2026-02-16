"""Compare Step 1 Run 1 reduction (used by Step 2 Run 2)"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING STEP 1 RUN 1 - REDUCTION")
print("(This is used by Step 2 Run 2 via timeStepequalHorizon)")
print("=" * 80)

# Python Step 1 Run 1
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 1 and entry.get('run') == 1:
        if 'initReach_tracking' in entry:
            py_entry = entry['initReach_tracking']
            break

# MATLAB Step 1 Run 1
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

mat_entry = None
for entry in matlab_log:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 1 and entry.run == 1:
            if hasattr(entry, 'initReach_tracking'):
                it = entry.initReach_tracking
                if isinstance(it, np.ndarray) and it.size > 0:
                    mat_entry = it[0]
                    break
                elif not isinstance(it, np.ndarray):
                    mat_entry = it
                    break

if py_entry is None:
    print("[ERROR] Could not find Python Step 1 Run 1")
    exit(1)
else:
    print("[OK] Found Python Step 1 Run 1")

if mat_entry is None:
    print("[ERROR] Could not find MATLAB Step 1 Run 1")
    exit(1)
else:
    print("[OK] Found MATLAB Step 1 Run 1")

# Compare generator counts
print("\n" + "-" * 80)
print("GENERATOR COUNTS - Rhom_tp REDUCTION")
print("-" * 80)

py_rhom_tp = py_entry.get('Rhom_tp_num_generators') if isinstance(py_entry, dict) else getattr(py_entry, 'Rhom_tp_num_generators', None)
py_rend_tp = py_entry.get('Rend_tp_num_generators') if isinstance(py_entry, dict) else getattr(py_entry, 'Rend_tp_num_generators', None)
py_redFactor = py_entry.get('redFactor') if isinstance(py_entry, dict) else getattr(py_entry, 'redFactor', None)

mat_rhom_tp = getattr(mat_entry, 'Rhom_tp_num_generators', None)
mat_rend_tp = getattr(mat_entry, 'Rend_tp_num_generators', None)
mat_redFactor = getattr(mat_entry, 'redFactor', None)

print(f"\nRhom_tp (before reduction):")
print(f"  Python: {py_rhom_tp}")
print(f"  MATLAB: {mat_rhom_tp}")
if py_rhom_tp == mat_rhom_tp:
    print(f"  [MATCH]")
else:
    print(f"  [MISMATCH]")

print(f"\nRend_tp (after reduction):")
print(f"  Python: {py_rend_tp}")
print(f"  MATLAB: {mat_rend_tp}")
if py_rend_tp == mat_rend_tp:
    print(f"  [MATCH]")
else:
    print(f"  [MISMATCH] - This is the divergence!")

print(f"\nredFactor:")
print(f"  Python: {py_redFactor}")
print(f"  MATLAB: {mat_redFactor}")
if py_redFactor == mat_redFactor:
    print(f"  [MATCH]")
else:
    print(f"  [MISMATCH]")

if py_rhom_tp and py_rend_tp and mat_rhom_tp and mat_rend_tp:
    py_reduced = py_rhom_tp - py_rend_tp
    mat_reduced = mat_rhom_tp - mat_rend_tp
    print(f"\nGenerators reduced:")
    print(f"  Python: {py_reduced} (from {py_rhom_tp} to {py_rend_tp})")
    print(f"  MATLAB: {mat_reduced} (from {mat_rhom_tp} to {mat_rend_tp})")
    if py_reduced != mat_reduced:
        print(f"  [MISMATCH] - Python reduces {py_reduced}, MATLAB reduces {mat_reduced}")
        print(f"  This is the root cause of the divergence!")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if py_rend_tp != mat_rend_tp:
    print("The divergence occurs in Step 1 Run 1's reduction of Rhom_tp.")
    print("This propagates to Step 2 Run 2 via timeStepequalHorizon path.")
    print("\nTo fix this, we need to compare:")
    print("  1. dHmax values (should be identical if redFactor matches)")
    print("  2. h_computed arrays (may differ due to numerical precision)")
    print("  3. redIdx values (determines how many generators are reduced)")
