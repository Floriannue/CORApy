"""Compare Step 2's Rtp (which becomes Step 3's Rstart)"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("COMPARING STEP 2's Rtp (becomes Step 3's Rstart)")
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

# Find Step 2 entries with initReach_tracking
step2 = 2
py_step2 = None
for e in python_upstream:
    if e.get('step') == step2 and 'initReach_tracking' in e:
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
                if get_ml_value(e, 'initReach_tracking') is not None:
                    ml_step2 = e
                    break

if py_step2 and ml_step2:
    print(f"\nStep {step2} - Rend.tp (from initReach_adaptive):\n")
    
    py_init = py_step2.get('initReach_tracking')
    ml_init = get_ml_value(ml_step2, 'initReach_tracking')
    
    if py_init and ml_init:
        # Compare Rend.tp (this becomes Rlintp, then Rtp, then next step's Rstart)
        py_rend_tp = py_init.get('Rend_tp_num_generators', 0)
        ml_rend_tp = get_ml_nested_value(ml_step2, 'initReach_tracking', 'Rend_tp_num_generators')
        if ml_rend_tp is not None and isinstance(ml_rend_tp, np.ndarray):
            ml_rend_tp = ml_rend_tp.item() if ml_rend_tp.size == 1 else ml_rend_tp[0,0] if ml_rend_tp.size > 0 else None
        
        print(f"Rend.tp (from initReach_adaptive):")
        print(f"  Python: {py_rend_tp} generators")
        print(f"  MATLAB: {ml_rend_tp} generators")
        if py_rend_tp != ml_rend_tp:
            print(f"  *** MISMATCH: {py_rend_tp} vs {ml_rend_tp} ***")
            print(f"  This Rend.tp becomes Rlintp, then Rtp, then Step 3's Rstart!")
        else:
            print(f"  Match!")
        
        # Also compare Rend.ti for completeness
        py_rend_ti = py_init.get('Rend_ti_num_generators', 0)
        ml_rend_ti = get_ml_nested_value(ml_step2, 'initReach_tracking', 'Rend_ti_num_generators')
        if ml_rend_ti is not None and isinstance(ml_rend_ti, np.ndarray):
            ml_rend_ti = ml_rend_ti.item() if ml_rend_ti.size == 1 else ml_rend_ti[0,0] if ml_rend_ti.size > 0 else None
        
        print(f"\nRend.ti (for reference):")
        print(f"  Python: {py_rend_ti} generators")
        print(f"  MATLAB: {ml_rend_ti} generators")
        if py_rend_ti != ml_rend_ti:
            print(f"  *** MISMATCH: {py_rend_ti} vs {ml_rend_ti} ***")
        else:
            print(f"  Match!")
    else:
        print("initReach_tracking not found")
else:
    print(f"Step {step2} entries not found")
