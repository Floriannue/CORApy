"""Compare generator counts and extract reduction info"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("DETAILED GENERATOR COUNT COMPARISON")
print("=" * 80)

# Python Step 2 Run 2
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        if 'initReach_tracking' in entry:
            py_entry = entry['initReach_tracking']
            break

# MATLAB - try to find any entry with initReach_tracking
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

print("\nPython Step 2 Run 2:")
if py_entry:
    print(f"  Rhom_tp_num_generators: {py_entry.get('Rhom_tp_num_generators', 'N/A')}")
    print(f"  Rend_tp_num_generators: {py_entry.get('Rend_tp_num_generators', 'N/A')}")
    print(f"  redFactor: {py_entry.get('redFactor', 'N/A')}")
    if py_entry.get('Rhom_tp_num_generators') and py_entry.get('Rend_tp_num_generators'):
        py_reduced = py_entry.get('Rhom_tp_num_generators') - py_entry.get('Rend_tp_num_generators')
        print(f"  Generators reduced: {py_reduced}")

print("\nMATLAB entries with initReach_tracking:")
matlab_entries = []
for entry in matlab_log:
    if hasattr(entry, 'initReach_tracking'):
        it = entry.initReach_tracking
        if isinstance(it, np.ndarray) and it.size > 0:
            matlab_entries.append((entry.step, entry.run, it[0]))
        elif not isinstance(it, np.ndarray):
            matlab_entries.append((entry.step, entry.run, it))

print(f"Found {len(matlab_entries)} MATLAB entries with initReach_tracking")

# Show first 10
for step, run, it in matlab_entries[:10]:
    rhom_tp = getattr(it, 'Rhom_tp_num_generators', None)
    rend_tp = getattr(it, 'Rend_tp_num_generators', None)
    if rhom_tp == 5:  # Match Python's input
        print(f"\n  Step {step}, Run {run}:")
        print(f"    Rhom_tp_num_generators: {rhom_tp}")
        print(f"    Rend_tp_num_generators: {rend_tp}")
        if rend_tp:
            reduced = rhom_tp - rend_tp
            print(f"    Generators reduced: {reduced}")
            if rend_tp == 4:
                print(f"    [MATCHES EXPECTED MATLAB RESULT]")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Python Step 2 Run 2: 5 generators -> 2 generators (reduced 3)")
print("MATLAB expected: 5 generators -> 4 generators (reduced 1)")
print("\nThe divergence is in the reduction algorithm.")
print("Both use the same input (5 generators) and redFactor (0.0005),")
print("but produce different outputs (2 vs 4 generators).")
print("\nThis suggests different h values or dHmax values in priv_reduceAdaptive.")
