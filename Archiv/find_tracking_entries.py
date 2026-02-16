"""Find entries with initReach_tracking in both Python and MATLAB"""
import pickle
import scipy.io
import numpy as np

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
py_entries = []
for entry in python_upstream:
    if isinstance(entry, dict):
        if 'initReach_tracking' in entry and entry.get('step') is not None:
            py_entries.append((entry.get('step'), entry.get('run')))
    elif hasattr(entry, 'initReach_tracking') and hasattr(entry, 'step'):
        py_entries.append((entry.step, entry.run))

# MATLAB
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

mat_entries = []
for entry in matlab_log:
    if hasattr(entry, 'initReach_tracking') and hasattr(entry, 'step'):
        it = entry.initReach_tracking
        # Check if it's not empty
        if isinstance(it, np.ndarray) and it.size > 0:
            mat_entries.append((entry.step, entry.run))
        elif not isinstance(it, np.ndarray):
            mat_entries.append((entry.step, entry.run))

print("Python entries with initReach_tracking:")
py_set = set(py_entries)
for step, run in sorted(py_set)[:20]:  # Show first 20
    print(f"  Step {step}, Run {run}")
print(f"  ... (total {len(py_set)} unique entries)")

print("\nMATLAB entries with non-empty initReach_tracking:")
mat_set = set(mat_entries)
for step, run in sorted(mat_set)[:20]:  # Show first 20
    print(f"  Step {step}, Run {run}")
print(f"  ... (total {len(mat_set)} unique entries)")

print("\nCommon entries (both have tracking):")
common = py_set & mat_set
for step, run in sorted(common)[:10]:  # Show first 10
    print(f"  Step {step}, Run {run}")
print(f"  ... (total {len(common)} common entries)")

if len(common) > 0:
    # Pick the first common entry
    step, run = sorted(common)[0]
    print(f"\nWill compare Step {step}, Run {run}")
