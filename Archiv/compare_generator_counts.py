"""Compare generator counts from Step 2 Run 2"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING GENERATOR COUNTS - STEP 2 RUN 2")
print("=" * 80)

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        py_entry = entry
        break

# MATLAB
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

mat_entry = None
for entry in matlab_log:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 2 and entry.run == 2:
            mat_entry = entry
            break

if py_entry is None:
    print("[ERROR] Could not find Python Step 2 Run 2")
    exit(1)

if mat_entry is None:
    print("[ERROR] Could not find MATLAB Step 2 Run 2")
    exit(1)

print("\nPython Step 2 Run 2:")
if 'initReach_tracking' in py_entry:
    it = py_entry['initReach_tracking']
    print(f"  Rend_tp_num_generators: {it.get('Rend_tp_num_generators', 'N/A')}")
    print(f"  Rend_ti_num_generators: {it.get('Rend_ti_num_generators', 'N/A')}")
    print(f"  Rhom_tp_num_generators: {it.get('Rhom_tp_num_generators', 'N/A')}")
    print(f"  Rhom_num_generators: {it.get('Rhom_num_generators', 'N/A')}")

if 'Rlintp_tracking' in py_entry:
    rlintp = py_entry['Rlintp_tracking']
    if isinstance(rlintp, dict):
        print(f"  Rlintp_num_generators: {rlintp.get('num_generators', 'N/A')}")

print("\nMATLAB Step 2 Run 2:")
if hasattr(mat_entry, 'Rlintp_tracking'):
    rlintp = mat_entry.Rlintp_tracking
    if isinstance(rlintp, np.ndarray) and rlintp.size > 0:
        print(f"  Rlintp_num_generators: {rlintp[0].num_generators if hasattr(rlintp[0], 'num_generators') else 'N/A'}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("The divergence in Rlintp (2 vs 4 generators) originates from initReach_adaptive's")
print("reduction of Rhom_tp to Rend_tp. To find the root cause, we need to compare:")
print("  1. The reduction parameters (dHmax, h_computed, redIdx)")
print("  2. The input zonotope (Rhom_tp) before reduction")
print("  3. The reduction algorithm execution")
print("\nHowever, MATLAB's initReach_tracking is empty for Step 2 Run 2.")
print("We need to re-run MATLAB with tracking enabled or check if there's a newer log file.")
