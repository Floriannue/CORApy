"""Compare Rstart (input to linReach_adaptive) between Python and MATLAB"""
import numpy as np
import pickle
import scipy.io as sio
import os

print("=" * 80)
print("COMPARING Rstart (Input to linReach_adaptive)")
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

def get_ml_nested_value(ml_obj, field, nested_field):
    parent = get_ml_value(ml_obj, field)
    if parent is None:
        return None
    if hasattr(parent, nested_field):
        val = getattr(parent, nested_field)
        if isinstance(val, np.ndarray):
            if val.size == 1:
                val = val.item()
            elif val.size > 0:
                val = val[0,0] if val.ndim > 0 else val.item()
        return val
    elif isinstance(parent, dict) and nested_field in parent:
        return parent[nested_field]
    return None

# Find Step 3 entries
step = 3
py_entry = None
for e in python_upstream:
    if e.get('step') == step and 'Rstart_tracking' in e:
        py_entry = e
        break

ml_entry = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step:
                ml_entry = e
                break

if py_entry and ml_entry:
    print(f"\nStep {step} - Rstart Comparison:\n")
    
    py_Rstart = py_entry.get('Rstart_tracking')
    ml_Rstart = get_ml_value(ml_entry, 'Rstart_tracking')
    
    if py_Rstart and ml_Rstart:
        py_num_gen = py_Rstart.get('num_generators', 0) if isinstance(py_Rstart, dict) else 0
        ml_num_gen = get_ml_nested_value(ml_entry, 'Rstart_tracking', 'num_generators')
        if ml_num_gen is not None and isinstance(ml_num_gen, np.ndarray):
            ml_num_gen = ml_num_gen.item() if ml_num_gen.size == 1 else ml_num_gen[0,0] if ml_num_gen.size > 0 else None
        
        print(f"Rstart generators:")
        print(f"  Python: {py_num_gen}")
        print(f"  MATLAB: {ml_num_gen}")
        if py_num_gen != ml_num_gen:
            print(f"  *** MISMATCH: {py_num_gen} vs {ml_num_gen} ***")
            print(f"  This is the ROOT CAUSE - Rstart differs at Step {step}!")
        else:
            print(f"  Match!")
    else:
        print("Rstart_tracking: NOT FOUND")
else:
    print(f"Step {step} entries not found")
