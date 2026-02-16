"""Check Step 1 Run 1 data since Step 2 Run 2 uses timeStepequalHorizon"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("CHECKING STEP 1 RUN 1 (used by Step 2 Run 2 via timeStepequalHorizon)")
print("=" * 80)

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 1 and entry.get('run') == 1:
        if 'initReach_tracking' in entry:
            py_entry = entry['initReach_tracking']
            break

# MATLAB
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

if py_entry is None:
    print("[ERROR] Could not find Python Step 1 Run 1")
else:
    print("[OK] Found Python Step 1 Run 1")
    print(f"  Rend_tp_num_generators: {py_entry.get('Rend_tp_num_generators', 'N/A')}")
    print(f"  Rhom_tp_num_generators: {py_entry.get('Rhom_tp_num_generators', 'N/A')}")

if mat_entry is None:
    print("[ERROR] Could not find MATLAB Step 1 Run 1")
else:
    print("[OK] Found MATLAB Step 1 Run 1")
    if hasattr(mat_entry, 'Rend_tp_num_generators'):
        print(f"  Rend_tp_num_generators: {mat_entry.Rend_tp_num_generators}")
    if hasattr(mat_entry, 'Rhom_tp_num_generators'):
        print(f"  Rhom_tp_num_generators: {mat_entry.Rhom_tp_num_generators}")

print("\n" + "=" * 80)
print("NOTE: Step 2 Run 2 uses timeStepequalHorizon path, which reuses Step 1 Run 1's")
print("initReach_adaptive results. So we should compare Step 1 Run 1, not Step 2 Run 2.")
print("=" * 80)
