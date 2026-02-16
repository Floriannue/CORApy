"""List all steps and runs in logs"""
import pickle
import scipy.io
import numpy as np

# Python log
print("Python log steps/runs:")
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

if isinstance(python_log, dict) and 'upstreamLog' in python_log:
    python_upstream = python_log['upstreamLog']
    steps_runs = set()
    for entry in python_upstream:
        if hasattr(entry, 'step') and hasattr(entry, 'run'):
            steps_runs.add((entry.step, entry.run))
        elif isinstance(entry, dict):
            if 'step' in entry and 'run' in entry:
                steps_runs.add((entry['step'], entry['run']))
    print(f"Found {len(steps_runs)} unique step/run combinations")
    for step, run in sorted(steps_runs):
        print(f"  Step {step}, Run {run}")

# MATLAB log
print("\nMATLAB log steps/runs:")
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

if isinstance(matlab_log, np.ndarray):
    steps_runs = set()
    for entry in matlab_log:
        if hasattr(entry, 'step') and hasattr(entry, 'run'):
            steps_runs.add((entry.step, entry.run))
    print(f"Found {len(steps_runs)} unique step/run combinations")
    for step, run in sorted(steps_runs):
        print(f"  Step {step}, Run {run}")
