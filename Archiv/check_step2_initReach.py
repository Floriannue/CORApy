"""Check if initReach_tracking exists in Step 2 entries."""
import pickle
import scipy.io
import numpy as np

# Load Python log
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

# Load MATLAB log
matlab_log = scipy.io.loadmat('upstream_matlab_log.mat', squeeze_me=True, struct_as_record=False)
matlab_upstream_log = matlab_log['upstreamLog']

# Find all Step 2 entries
python_step2_entries = []
for entry in python_log.get('upstreamLog', []):
    if entry.get('step') == 2:
        python_step2_entries.append(entry)

matlab_step2_entries = []
for entry in matlab_upstream_log:
    if hasattr(entry, 'step') and entry.step == 2:
        matlab_step2_entries.append(entry)

print(f"Python Step 2 entries: {len(python_step2_entries)}")
for i, entry in enumerate(python_step2_entries):
    run = entry.get('run')
    has_initReach = 'initReach_tracking' in entry
    print(f"  Entry {i+1}: run={run}, has initReach_tracking={has_initReach}")

print()
print(f"MATLAB Step 2 entries: {len(matlab_step2_entries)}")
for i, entry in enumerate(matlab_step2_entries):
    run = entry.run if hasattr(entry, 'run') else None
    if isinstance(run, np.ndarray):
        run = run.item() if run.size == 1 else run
    has_initReach = hasattr(entry, 'initReach_tracking')
    if has_initReach:
        initReach = entry.initReach_tracking
        if isinstance(initReach, np.ndarray) and initReach.size == 0:
            has_initReach = False
    print(f"  Entry {i+1}: run={run}, has initReach_tracking={has_initReach}")
