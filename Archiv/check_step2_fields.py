"""Check what fields Step 2 entries have in the logs"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("CHECKING STEP 2 FIELDS IN LOGS")
print("=" * 80)

# Load logs
python_file = 'upstream_python_log.pkl'
with open(python_file, 'rb') as f:
    python_data = pickle.load(f)
python_upstream = python_data.get('upstreamLog', [])

matlab_file = 'upstream_matlab_log.mat'
matlab_data = sio.loadmat(matlab_file, squeeze_me=False, struct_as_record=False)
matlab_upstream = matlab_data.get('upstreamLog', [])

def get_ml_value(ml_obj, field):
    if hasattr(ml_obj, 'dtype') and ml_obj.dtype.names and field in ml_obj.dtype.names:
        val = ml_obj[field]
        if isinstance(val, np.ndarray) and val.size > 0:
            if val.dtype == object and val.size == 1:
                val = val.item()
            elif hasattr(val, 'dtype') and val.dtype == object:
                val = val.item() if val.size == 1 else val[0,0]
        return val
    elif hasattr(ml_obj, field):
        val = getattr(ml_obj, field)
        if isinstance(val, np.ndarray) and val.size > 0:
            if val.dtype == object or (hasattr(val, 'dtype') and val.dtype == object):
                val = val.item() if val.size == 1 else val[0,0]
            elif val.size == 1:
                val = val.item()
        return val
    return None

# Find all Step 2 entries
step2 = 2
py_step2_entries = [e for e in python_upstream if e.get('step') == step2]
ml_step2_entries = []

if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step2:
                ml_step2_entries.append((i, e))

print(f"\nPython Step 2 entries: {len(py_step2_entries)}")
for i, e in enumerate(py_step2_entries):
    print(f"\n  Entry {i+1}:")
    print(f"    Step: {e.get('step')}")
    print(f"    Run: {e.get('run')}")
    print(f"    Keys: {list(e.keys())[:20]}")

print(f"\nMATLAB Step 2 entries: {len(ml_step2_entries)}")
for i, (idx, e) in enumerate(ml_step2_entries):
    print(f"\n  Entry {i+1} (index {idx}):")
    step_val = get_ml_value(e, 'step')
    run_val = get_ml_value(e, 'run')
    print(f"    Step: {step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val}")
    print(f"    Run: {run_val.item() if isinstance(run_val, np.ndarray) and run_val.size == 1 else run_val}")
    if hasattr(e, 'dtype') and e.dtype.names:
        print(f"    Fields: {list(e.dtype.names)[:20]}")

# Check if any Step 2 entry has Rtp_final_tracking
print("\n" + "=" * 80)
print("CHECKING FOR Rtp_final_tracking IN STEP 2:")
print("=" * 80)

py_has_rtp = any('Rtp_final_tracking' in e for e in py_step2_entries)
print(f"Python: {'Has Rtp_final_tracking' if py_has_rtp else 'No Rtp_final_tracking'}")

ml_has_rtp = False
for idx, e in ml_step2_entries:
    if hasattr(e, 'dtype') and e.dtype.names and 'Rtp_final_tracking' in e.dtype.names:
        ml_has_rtp = True
        break
    elif hasattr(e, 'Rtp_final_tracking'):
        ml_has_rtp = True
        break
print(f"MATLAB: {'Has Rtp_final_tracking' if ml_has_rtp else 'No Rtp_final_tracking'}")

# Check for Rlintp_tracking and Rerror_tracking
print("\n" + "=" * 80)
print("CHECKING FOR Rlintp_tracking AND Rerror_tracking IN STEP 2:")
print("=" * 80)

py_has_rlintp = any('Rlintp_tracking' in e for e in py_step2_entries)
py_has_rerror = any('Rerror_tracking' in e for e in py_step2_entries)
print(f"Python: Rlintp_tracking={'Yes' if py_has_rlintp else 'No'}, Rerror_tracking={'Yes' if py_has_rerror else 'No'}")

ml_has_rlintp = False
ml_has_rerror = False
for idx, e in ml_step2_entries:
    if hasattr(e, 'dtype') and e.dtype.names:
        if 'Rlintp_tracking' in e.dtype.names:
            ml_has_rlintp = True
        if 'Rerror_tracking' in e.dtype.names:
            ml_has_rerror = True
    if hasattr(e, 'Rlintp_tracking'):
        ml_has_rlintp = True
    if hasattr(e, 'Rerror_tracking'):
        ml_has_rerror = True
print(f"MATLAB: Rlintp_tracking={'Yes' if ml_has_rlintp else 'No'}, Rerror_tracking={'Yes' if ml_has_rerror else 'No'}")
