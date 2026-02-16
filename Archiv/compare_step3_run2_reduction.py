"""Compare Step 3 Run 2 reduction parameters"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING STEP 3 RUN 2 - REDUCTION PARAMETERS")
print("=" * 80)

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 3 and entry.get('run') == 2:
        if 'initReach_tracking' in entry:
            py_entry = entry['initReach_tracking']
            break

# MATLAB
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

mat_entry = None
for entry in matlab_log:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 3 and entry.run == 2:
            if hasattr(entry, 'initReach_tracking'):
                it = entry.initReach_tracking
                if isinstance(it, np.ndarray) and it.size > 0:
                    mat_entry = it[0]
                    break

if py_entry is None:
    print("[ERROR] Could not find Python Step 3 Run 2")
    exit(1)

if mat_entry is None:
    print("[ERROR] Could not find MATLAB Step 3 Run 2")
    exit(1)

print("\nPython Step 3 Run 2:")
print(f"  Rend_tp_num_generators: {py_entry.get('Rend_tp_num_generators', 'N/A')}")
print(f"  Rend_ti_num_generators: {py_entry.get('Rend_ti_num_generators', 'N/A')}")
print(f"  Rhom_tp_num_generators: {py_entry.get('Rhom_tp_num_generators', 'N/A')}")
print(f"  Rhom_num_generators: {py_entry.get('Rhom_num_generators', 'N/A')}")

print("\nMATLAB Step 3 Run 2:")
if hasattr(mat_entry, 'Rend_tp_num_generators'):
    print(f"  Rend_tp_num_generators: {mat_entry.Rend_tp_num_generators}")
if hasattr(mat_entry, 'Rend_ti_num_generators'):
    print(f"  Rend_ti_num_generators: {mat_entry.Rend_ti_num_generators}")
if hasattr(mat_entry, 'Rhom_tp_num_generators'):
    print(f"  Rhom_tp_num_generators: {mat_entry.Rhom_tp_num_generators}")
if hasattr(mat_entry, 'Rhom_num_generators'):
    print(f"  Rhom_num_generators: {mat_entry.Rhom_num_generators}")

# Check if there's reduction_tp data
print("\n" + "-" * 80)
print("REDUCTION_TP COMPARISON")
print("-" * 80)

py_red_tp = py_entry.get('reduction_tp') if isinstance(py_entry, dict) else None
mat_red_tp = getattr(mat_entry, 'reduction_tp', None)

if py_red_tp is None:
    print("Python: No reduction_tp data")
else:
    print("Python reduction_tp:")
    if isinstance(py_red_tp, dict):
        for key in ['dHmax', 'h_computed', 'redIdx', 'nrG', 'last0Idx']:
            if key in py_red_tp:
                val = py_red_tp[key]
                if isinstance(val, (list, np.ndarray)):
                    if len(val) <= 10:
                        print(f"  {key}: {val}")
                    else:
                        print(f"  {key}: array of length {len(val)}")
                else:
                    print(f"  {key}: {val}")

if mat_red_tp is None:
    print("MATLAB: No reduction_tp data")
else:
    print("MATLAB reduction_tp:")
    if isinstance(mat_red_tp, np.ndarray) and mat_red_tp.size > 0:
        mat_red_tp = mat_red_tp[0]
    if hasattr(mat_red_tp, '_fieldnames'):
        for key in ['dHmax', 'h_computed', 'redIdx', 'nrG', 'last0Idx']:
            if hasattr(mat_red_tp, key):
                val = getattr(mat_red_tp, key)
                if isinstance(val, np.ndarray):
                    if val.size <= 10:
                        print(f"  {key}: {val}")
                    else:
                        print(f"  {key}: array of length {val.size}")
                else:
                    print(f"  {key}: {val}")
