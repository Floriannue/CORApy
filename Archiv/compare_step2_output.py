"""Compare Step 2 output (which becomes Rstart for Step 3)"""
import numpy as np
import pickle
import scipy.io as sio
import os

print("=" * 80)
print("COMPARING STEP 2 OUTPUT (becomes Rstart for Step 3)")
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

# Find Step 2 entries
step2 = 2
step3 = 3

# Step 2 - Rstart (should be Step 1's output)
print("\n=== STEP 2 - Rstart (input to linReach_adaptive) ===")
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
                ml_step2 = e
                break

if py_step2 and ml_step2:
    py_Rstart = py_step2.get('Rstart_tracking')
    ml_Rstart = get_ml_value(ml_step2, 'Rstart_tracking')
    if py_Rstart and ml_Rstart:
        py_num = py_Rstart.get('num_generators', 0) if isinstance(py_Rstart, dict) else 0
        ml_num = get_ml_nested_value(ml_step2, 'Rstart_tracking', 'num_generators')
        if ml_num is not None and isinstance(ml_num, np.ndarray):
            ml_num = ml_num.item() if ml_num.size == 1 else ml_num[0,0] if ml_num.size > 0 else None
        print(f"Step 2 Rstart: Python {py_num} gens, MATLAB {ml_num} gens")
        if py_num != ml_num:
            print(f"  *** MISMATCH ***")
        else:
            print(f"  Match!")

# Step 3 - Rstart (should be Step 2's output)
print("\n=== STEP 3 - Rstart (input to linReach_adaptive) ===")
py_step3 = None
for e in python_upstream:
    if e.get('step') == step3 and 'Rstart_tracking' in e:
        py_step3 = e
        break

ml_step3 = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step3:
                ml_step3 = e
                break

if py_step3 and ml_step3:
    py_Rstart = py_step3.get('Rstart_tracking')
    ml_Rstart = get_ml_value(ml_step3, 'Rstart_tracking')
    if py_Rstart and ml_Rstart:
        py_num = py_Rstart.get('num_generators', 0) if isinstance(py_Rstart, dict) else 0
        ml_num = get_ml_nested_value(ml_step3, 'Rstart_tracking', 'num_generators')
        if ml_num is not None and isinstance(ml_num, np.ndarray):
            ml_num = ml_num.item() if ml_num.size == 1 else ml_num[0,0] if ml_num.size > 0 else None
        print(f"Step 3 Rstart: Python {py_num} gens, MATLAB {ml_num} gens")
        if py_num != ml_num:
            print(f"  *** MISMATCH - This is the problem! ***")
        else:
            print(f"  Match!")

# Check Step 2's final output (Rti/Rtp) - this should become Step 3's Rstart
print("\n=== STEP 2 - Final Output (should become Step 3's Rstart) ===")
# We need to check what Step 2's final Rti/Rtp is
# This would be in the reachable set, not in the upstream log
# Let me check if there's tracking for the final output

# Check initReach_tracking for Step 2 - Rend.ti should be the output
py_step2_init = None
for e in python_upstream:
    if e.get('step') == step2 and 'initReach_tracking' in e:
        py_step2_init = e
        break

ml_step2_init = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step2:
                if get_ml_value(e, 'initReach_tracking') is not None:
                    ml_step2_init = e
                    break

if py_step2_init and ml_step2_init:
    py_init = py_step2_init.get('initReach_tracking')
    ml_init = get_ml_value(ml_step2_init, 'initReach_tracking')
    if py_init and ml_init:
        py_rend_ti = py_init.get('Rend_ti_num_generators', 0)
        ml_rend_ti = get_ml_nested_value(ml_step2_init, 'initReach_tracking', 'Rend_ti_num_generators')
        if ml_rend_ti is not None and isinstance(ml_rend_ti, np.ndarray):
            ml_rend_ti = ml_rend_ti.item() if ml_rend_ti.size == 1 else ml_rend_ti[0,0] if ml_rend_ti.size > 0 else None
        print(f"Step 2 Rend.ti (final output): Python {py_rend_ti} gens, MATLAB {ml_rend_ti} gens")
        if py_rend_ti != ml_rend_ti:
            print(f"  *** MISMATCH - Step 2 output differs! ***")
        else:
            print(f"  Match!")
