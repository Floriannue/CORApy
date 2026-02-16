"""Compare Step 2's Rtp components to find where the 2-generator difference comes from"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("COMPARING STEP 2's Rtp COMPONENTS")
print("=" * 80)
print("Rtp = Rlintp + nlnsys.linError.p.x + Rerror")
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

# Find Step 2 entries
step2 = 2

# 1. Compare Rend.tp from initReach_adaptive (this becomes Rlintp)
print("\n1. Rend.tp from initReach_adaptive (becomes Rlintp):")
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
    ml_init_field = get_ml_value(ml_step2_init, 'initReach_tracking')
    if isinstance(ml_init_field, np.ndarray) and ml_init_field.size > 0:
        ml_init = ml_init_field.item() if ml_init_field.size == 1 else ml_init_field[0,0]
    else:
        ml_init = ml_init_field
    
    if py_init and ml_init:
        py_rend_tp = py_init.get('Rend_tp_num_generators', 0)
        ml_rend_tp_val = getattr(ml_init, 'Rend_tp_num_generators', None) if ml_init else None
        if ml_rend_tp_val is not None:
            if isinstance(ml_rend_tp_val, np.ndarray):
                ml_rend_tp = ml_rend_tp_val.item() if ml_rend_tp_val.size == 1 else ml_rend_tp_val[0,0] if ml_rend_tp_val.size > 0 else None
            else:
                ml_rend_tp = ml_rend_tp_val
        else:
            ml_rend_tp = None
        
        print(f"   Python: {py_rend_tp} generators")
        print(f"   MATLAB: {ml_rend_tp} generators")
        if py_rend_tp != ml_rend_tp:
            print(f"   *** MISMATCH: {py_rend_tp} vs {ml_rend_tp} ***")
        else:
            print(f"   Match!")

# 2. Compare Rtp_final_tracking (final Rtp from linReach_adaptive, before adding Rerror)
print("\n2. Rtp_final_tracking (from linReach_adaptive, before adding Rerror):")
py_step2_rtp = None
for e in python_upstream:
    if e.get('step') == step2 and 'Rtp_final_tracking' in e:
        py_step2_rtp = e
        break

ml_step2_rtp = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step2:
                if get_ml_value(e, 'Rtp_final_tracking') is not None:
                    ml_step2_rtp = e
                    break

if py_step2_rtp and ml_step2_rtp:
    py_rtp_final = py_step2_rtp.get('Rtp_final_tracking')
    ml_rtp_final_field = get_ml_value(ml_step2_rtp, 'Rtp_final_tracking')
    if isinstance(ml_rtp_final_field, np.ndarray) and ml_rtp_final_field.size > 0:
        ml_rtp_final = ml_rtp_final_field.item() if ml_rtp_final_field.size == 1 else ml_rtp_final_field[0,0]
    else:
        ml_rtp_final = ml_rtp_final_field
    
    if py_rtp_final and ml_rtp_final:
        py_num = py_rtp_final.get('num_generators', 0) if isinstance(py_rtp_final, dict) else 0
        ml_num = None
        if ml_rtp_final is not None:
            ml_num_val = getattr(ml_rtp_final, 'num_generators', None)
            if ml_num_val is not None:
                if isinstance(ml_num_val, np.ndarray):
                    ml_num = ml_num_val.item() if ml_num_val.size == 1 else ml_num_val[0,0] if ml_num_val.size > 0 else None
                else:
                    ml_num = ml_num_val
        
        print(f"   Python: {py_num} generators")
        print(f"   MATLAB: {ml_num} generators")
        if py_num != ml_num:
            print(f"   *** MISMATCH: {py_num} vs {ml_num} ***")
            print(f"   Difference: {abs(py_num - ml_num)} generators")
        else:
            print(f"   Match!")

# 3. Compare Rtp before reduction in reach_adaptive (this includes Rerror)
print("\n3. Rtp BEFORE reduction in reach_adaptive (includes Rerror):")
python_Rtp_tracking = python_data.get('Rtp_tracking', {})
py_step2_rtp_reach = python_Rtp_tracking.get(step2 + 1, {})  # Step 2's Rtp becomes Step 3's Rstart
py_before = py_step2_rtp_reach.get('before', {})

matlab_Rtp_tracking = matlab_data.get('Rtp_tracking', None)
ml_step2_rtp_reach = None
if matlab_Rtp_tracking is not None:
    if isinstance(matlab_Rtp_tracking, np.ndarray) and matlab_Rtp_tracking.size > 0:
        rtp_obj = matlab_Rtp_tracking.item() if matlab_Rtp_tracking.size == 1 else matlab_Rtp_tracking[0,0]
        step_field = f'step_{step2 + 1}'
        if hasattr(rtp_obj, step_field):
            ml_step2_field = getattr(rtp_obj, step_field)
            if isinstance(ml_step2_field, np.ndarray) and ml_step2_field.size > 0:
                ml_step2_rtp_reach = ml_step2_field.item() if ml_step2_field.size == 1 else ml_step2_field[0,0]
            else:
                ml_step2_rtp_reach = ml_step2_field

if py_before and ml_step2_rtp_reach:
    py_num = py_before.get('num_generators', 0)
    ml_before_field = getattr(ml_step2_rtp_reach, 'before', None) if ml_step2_rtp_reach else None
    if isinstance(ml_before_field, np.ndarray) and ml_before_field.size > 0:
        ml_before = ml_before_field.item() if ml_before_field.size == 1 else ml_before_field[0,0]
    else:
        ml_before = ml_before_field
    
    ml_num = None
    if ml_before is not None:
        ml_num_val = getattr(ml_before, 'num_generators', None)
        if ml_num_val is not None:
            if isinstance(ml_num_val, np.ndarray):
                ml_num = ml_num_val.item() if ml_num_val.size == 1 else ml_num_val[0,0] if ml_num_val.size > 0 else None
            else:
                ml_num = ml_num_val
    
    print(f"   Python: {py_num} generators")
    print(f"   MATLAB: {ml_num} generators")
    if py_num != ml_num:
        print(f"   *** MISMATCH: {py_num} vs {ml_num} ***")
        print(f"   Difference: {abs(py_num - ml_num)} generators")
    else:
        print(f"   Match!")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("If Rend.tp matches but Rtp_final_tracking differs, the issue is in the translation")
print("If Rtp_final_tracking matches but Rtp before reduction differs, the issue is in Rerror addition")
