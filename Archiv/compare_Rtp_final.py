"""Compare final Rtp from linReach_adaptive (before reduction in reach_adaptive)"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("COMPARING FINAL Rtp FROM linReach_adaptive (Step 2)")
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

# Find Step 2 entries with Rtp_final_tracking
step2 = 2
py_step2 = None
for e in python_upstream:
    if e.get('step') == step2 and 'Rtp_final_tracking' in e:
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
                if get_ml_value(e, 'Rtp_final_tracking') is not None:
                    ml_step2 = e
                    break

if py_step2 and ml_step2:
    print(f"\nStep {step2} - Final Rtp (from linReach_adaptive, before reduction in reach_adaptive):\n")
    
    py_Rtp = py_step2.get('Rtp_final_tracking')
    ml_Rtp = get_ml_value(ml_step2, 'Rtp_final_tracking')
    
    if py_Rtp and ml_Rtp:
        py_num = py_Rtp.get('num_generators', 0) if isinstance(py_Rtp, dict) else 0
        ml_num = get_ml_nested_value(ml_step2, 'Rtp_final_tracking', 'num_generators')
        if ml_num is not None and isinstance(ml_num, np.ndarray):
            ml_num = ml_num.item() if ml_num.size == 1 else ml_num[0,0] if ml_num.size > 0 else None
        
        print(f"Final Rtp (before reduction in reach_adaptive):")
        print(f"  Python: {py_num} generators")
        print(f"  MATLAB: {ml_num} generators")
        if py_num != ml_num:
            print(f"  *** MISMATCH: {py_num} vs {ml_num} ***")
            print(f"  This Rtp is reduced in reach_adaptive to become Step 3's Rstart!")
            print(f"  The reduction uses redFactor={py_Rtp.get('redFactor', 'N/A') if isinstance(py_Rtp, dict) else 'N/A'}")
        else:
            print(f"  Match!")
    else:
        print("Rtp_final_tracking not found")
else:
    print(f"Step {step2} entries with Rtp_final_tracking not found")
    if not py_step2:
        print("  Python: Missing")
    if not ml_step2:
        print("  MATLAB: Missing")
