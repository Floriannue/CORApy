"""Compare Step 2 Run 2 reduction - compare what's actually available"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING STEP 2 RUN 2 - REDUCTION DATA")
print("=" * 80)

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        if 'initReach_tracking' in entry:
            py_entry = entry['initReach_tracking']
            break

# MATLAB - try Step 2 Run 2 first, then Step 1 Run 1 as fallback
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

mat_entry = None
mat_source = None
# Try Step 2 Run 2 first
for entry in matlab_log:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 2 and entry.run == 2:
            if hasattr(entry, 'initReach_tracking'):
                it = entry.initReach_tracking
                if isinstance(it, np.ndarray) and it.size > 0:
                    mat_entry = it[0]
                    mat_source = "Step 2 Run 2"
                    break
                elif not isinstance(it, np.ndarray):
                    mat_entry = it
                    mat_source = "Step 2 Run 2"
                    break

# Fallback to Step 1 Run 1 (used by timeStepequalHorizon)
if mat_entry is None:
    for entry in matlab_log:
        if hasattr(entry, 'step') and hasattr(entry, 'run'):
            if entry.step == 1 and entry.run == 1:
                if hasattr(entry, 'initReach_tracking'):
                    it = entry.initReach_tracking
                    if isinstance(it, np.ndarray) and it.size > 0:
                        mat_entry = it[0]
                        mat_source = "Step 1 Run 1 (used by Step 2 Run 2)"
                        break
                    elif not isinstance(it, np.ndarray):
                        mat_entry = it
                        mat_source = "Step 1 Run 1 (used by Step 2 Run 2)"
                        break

if py_entry is None:
    print("[ERROR] Could not find Python Step 2 Run 2 initReach_tracking")
    exit(1)
else:
    print("[OK] Found Python Step 2 Run 2 entry")

if mat_entry is None:
    print("[ERROR] Could not find MATLAB entry (tried Step 2 Run 2 and Step 1 Run 1)")
    exit(1)
else:
    print(f"[OK] Found MATLAB entry from {mat_source}")

# Compare generator counts
print("\n" + "-" * 80)
print("GENERATOR COUNTS COMPARISON")
print("-" * 80)

# Python
py_rhom_tp = py_entry.get('Rhom_tp_num_generators') if isinstance(py_entry, dict) else getattr(py_entry, 'Rhom_tp_num_generators', None)
py_rend_tp = py_entry.get('Rend_tp_num_generators') if isinstance(py_entry, dict) else getattr(py_entry, 'Rend_tp_num_generators', None)

# MATLAB
mat_rhom_tp = getattr(mat_entry, 'Rhom_tp_num_generators', None)
mat_rend_tp = getattr(mat_entry, 'Rend_tp_num_generators', None)

print(f"\nRhom_tp (before reduction):")
print(f"  Python: {py_rhom_tp}")
print(f"  MATLAB: {mat_rhom_tp}")
if py_rhom_tp == mat_rhom_tp:
    print(f"  [MATCH]")
else:
    print(f"  [MISMATCH] - Difference: {abs(py_rhom_tp - mat_rhom_tp) if py_rhom_tp and mat_rhom_tp else 'N/A'}")

print(f"\nRend_tp (after reduction):")
print(f"  Python: {py_rend_tp}")
print(f"  MATLAB: {mat_rend_tp}")
if py_rend_tp == mat_rend_tp:
    print(f"  [MATCH]")
else:
    print(f"  [MISMATCH] - Difference: {abs(py_rend_tp - mat_rend_tp) if py_rend_tp and mat_rend_tp else 'N/A'}")

if py_rhom_tp and py_rend_tp and mat_rhom_tp and mat_rend_tp:
    py_reduced = py_rhom_tp - py_rend_tp
    mat_reduced = mat_rhom_tp - mat_rend_tp
    print(f"\nGenerators reduced:")
    print(f"  Python: {py_reduced} (from {py_rhom_tp} to {py_rend_tp})")
    print(f"  MATLAB: {mat_reduced} (from {mat_rhom_tp} to {mat_rend_tp})")
    print(f"  Difference: {abs(py_reduced - mat_reduced)} generators")

# Check for reduction_tp data
print("\n" + "-" * 80)
print("CHECKING FOR REDUCTION PARAMETERS")
print("-" * 80)

# Python
if isinstance(py_entry, dict):
    py_red_tp = py_entry.get('reduction_tp')
    py_keys = list(py_entry.keys())
else:
    py_red_tp = getattr(py_entry, 'reduction_tp', None)
    py_keys = [attr for attr in dir(py_entry) if not attr.startswith('_')]

print(f"\nPython entry keys/fields: {py_keys[:20]}...")  # Show first 20

# MATLAB
if hasattr(mat_entry, '_fieldnames'):
    mat_keys = mat_entry._fieldnames
    mat_red_tp = getattr(mat_entry, 'reduction_tp', None) if 'reduction_tp' in mat_keys else None
else:
    mat_keys = [attr for attr in dir(mat_entry) if not attr.startswith('_')]
    mat_red_tp = getattr(mat_entry, 'reduction_tp', None)

print(f"MATLAB entry fields: {mat_keys[:20]}...")  # Show first 20

if py_red_tp is None:
    print("\n[INFO] Python does not have reduction_tp field")
    print("       Reduction parameters may be stored elsewhere or not tracked")
else:
    print("\n[OK] Python has reduction_tp field")

if mat_red_tp is None:
    print("[INFO] MATLAB does not have reduction_tp field")
    print("       Reduction parameters may be stored elsewhere or not tracked")
else:
    print("[OK] MATLAB has reduction_tp field")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("The divergence in generator counts shows:")
print(f"  Python reduces {py_reduced if py_rhom_tp and py_rend_tp else 'N/A'} generators")
print(f"  MATLAB reduces {mat_reduced if mat_rhom_tp and mat_rend_tp else 'N/A'} generators")
print("\nTo find the root cause, we need to compare:")
print("  1. dHmax values")
print("  2. h_computed arrays")
print("  3. redIdx values")
print("\nThese may be stored in reduction_tp or need to be extracted from reduction debug files.")
