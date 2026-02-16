"""Compare Step 2's Rstart (input to linReach_adaptive)"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("COMPARING STEP 2's Rstart (Input to linReach_adaptive)")
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
    if isinstance(parent, np.ndarray) and parent.size > 0:
        parent = parent.item() if parent.size == 1 else parent[0,0]
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

# Find Step 2 entries with Rstart_tracking
step2 = 2
py_step2 = None
for e in python_upstream:
    if e.get('step') == step2 and 'Rstart_tracking' in e:
        py_step2 = e
        break

ml_step2 = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step2:
                if get_ml_value(e, 'Rstart_tracking') is not None:
                    ml_step2 = e
                    break

if py_step2 and ml_step2:
    print(f"\nStep {step2} - Rstart (input to linReach_adaptive):\n")
    
    py_Rstart = py_step2.get('Rstart_tracking')
    ml_Rstart_field = get_ml_value(ml_step2, 'Rstart_tracking')
    if isinstance(ml_Rstart_field, np.ndarray) and ml_Rstart_field.size > 0:
        ml_Rstart = ml_Rstart_field.item() if ml_Rstart_field.size == 1 else ml_Rstart_field[0,0]
    else:
        ml_Rstart = ml_Rstart_field
    
    if py_Rstart and ml_Rstart:
        py_num = py_Rstart.get('num_generators', 0) if isinstance(py_Rstart, dict) else 0
        ml_num = None
        if ml_Rstart is not None:
            ml_num_val = getattr(ml_Rstart, 'num_generators', None)
            if ml_num_val is not None:
                if isinstance(ml_num_val, np.ndarray):
                    ml_num = ml_num_val.item() if ml_num_val.size == 1 else ml_num_val[0,0] if ml_num_val.size > 0 else None
                else:
                    ml_num = ml_num_val
        
        print(f"Rstart generators:")
        print(f"  Python: {py_num}")
        print(f"  MATLAB: {ml_num}")
        if py_num != ml_num:
            print(f"  *** MISMATCH: {py_num} vs {ml_num} ***")
            print(f"  This is the ROOT CAUSE - Step 2's Rstart differs!")
        else:
            print(f"  Match!")
    else:
        print("Rstart_tracking not found")
else:
    print(f"Step {step2} entries not found")
