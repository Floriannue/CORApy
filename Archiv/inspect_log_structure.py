"""Inspect log structure"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("INSPECTING LOG STRUCTURES")
print("=" * 80)

# Python log
print("\nPython log:")
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

print(f"Type: {type(python_log)}")
print(f"Keys: {list(python_log.keys()) if isinstance(python_log, dict) else 'N/A'}")

if isinstance(python_log, dict) and len(python_log) > 0:
    first_key = list(python_log.keys())[0]
    first_entry = python_log[first_key]
    print(f"First key: {first_key}")
elif isinstance(python_log, (list, np.ndarray)) and len(python_log) > 0:
    first_entry = python_log[0]
else:
    first_entry = None

if first_entry is not None:
    print(f"First entry type: {type(first_entry)}")
    print(f"First entry attributes: {dir(first_entry)}")
    if hasattr(first_entry, 'step'):
        print(f"First entry step: {first_entry.step}")
    if hasattr(first_entry, 'run'):
        print(f"First entry run: {first_entry.run}")
    if hasattr(first_entry, 'initReach_tracking'):
        print(f"Has initReach_tracking: True")
        print(f"initReach_tracking type: {type(first_entry.initReach_tracking)}")
        print(f"initReach_tracking attributes: {dir(first_entry.initReach_tracking)}")

# MATLAB log
print("\nMATLAB log:")
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

print(f"Type: {type(matlab_log)}")
print(f"Shape: {matlab_log.shape if hasattr(matlab_log, 'shape') else 'N/A'}")

if matlab_log.size > 0:
    first_entry = matlab_log[0]
    print(f"First entry type: {type(first_entry)}")
    print(f"First entry attributes: {dir(first_entry)}")
    if hasattr(first_entry, 'step'):
        print(f"First entry step: {first_entry.step}")
    if hasattr(first_entry, 'run'):
        print(f"First entry run: {first_entry.run}")
    if hasattr(first_entry, 'initReach_tracking'):
        print(f"Has initReach_tracking: True")
        print(f"initReach_tracking type: {type(first_entry.initReach_tracking)}")
        print(f"initReach_tracking attributes: {dir(first_entry.initReach_tracking)}")
