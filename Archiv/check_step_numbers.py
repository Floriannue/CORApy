"""Check step numbers in logs."""
import pickle
import scipy.io
import numpy as np

# Load Python log
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

# Load MATLAB log
matlab_log = scipy.io.loadmat('upstream_matlab_log.mat', squeeze_me=True, struct_as_record=False)
matlab_upstream_log = matlab_log['upstreamLog']

# Find Step 2 entries
python_step2 = None
for entry in python_log.get('upstreamLog', []):
    if entry.get('step') == 2:
        python_step2 = entry
        print(f"Python Step 2: run={entry.get('run')}, has initReach_tracking={'initReach_tracking' in entry}")

# Find all MATLAB entries with step 2
print("\nMATLAB entries with step 2:")
for i, entry in enumerate(matlab_upstream_log):
    if hasattr(entry, 'step'):
        step = entry.step
        if isinstance(step, np.ndarray):
            step = step.item() if step.size == 1 else step
        if step == 2:
            run = entry.run if hasattr(entry, 'run') else None
            if isinstance(run, np.ndarray):
                run = run.item() if run.size == 1 else run
            has_initReach = hasattr(entry, 'initReach_tracking')
            if has_initReach:
                initReach = entry.initReach_tracking
                if isinstance(initReach, np.ndarray) and initReach.size == 0:
                    has_initReach = False
            print(f"  Entry {i}: step={step}, run={run}, has initReach_tracking={has_initReach}")
