"""Check which code path each step uses"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("CHECKING CODE PATH FOR ALL STEPS")
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

# Collect code path info for all steps
steps_info = {}
for step in range(1, 11):  # Check steps 1-10
    py_entry = None
    for e in python_upstream:
        if e.get('step') == step and 'Rtp_final_tracking' in e:
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
                    if hasattr(e, 'dtype') and e.dtype.names and 'Rtp_final_tracking' in e.dtype.names:
                        ml_entry = e
                        break
    
    if py_entry or ml_entry:
        info = {'step': step}
        
        if py_entry:
            py_rtp_final = py_entry.get('Rtp_final_tracking')
            if py_rtp_final:
                info['py_timeStepequalHorizon'] = py_rtp_final.get('timeStepequalHorizon_used', None) if isinstance(py_rtp_final, dict) else None
                info['py_rtp_num'] = py_rtp_final.get('num_generators', 0) if isinstance(py_rtp_final, dict) else 0
        
        if ml_entry:
            ml_rtp_final_field = get_ml_value(ml_entry, 'Rtp_final_tracking')
            if isinstance(ml_rtp_final_field, np.ndarray) and ml_rtp_final_field.size > 0:
                ml_rtp_final = ml_rtp_final_field.item() if ml_rtp_final_field.size == 1 else ml_rtp_final_field[0,0]
            else:
                ml_rtp_final = ml_rtp_final_field
            
            if ml_rtp_final is not None:
                ml_timeStepequalHorizon_val = getattr(ml_rtp_final, 'timeStepequalHorizon_used', None)
                if ml_timeStepequalHorizon_val is not None:
                    if isinstance(ml_timeStepequalHorizon_val, np.ndarray):
                        info['ml_timeStepequalHorizon'] = ml_timeStepequalHorizon_val.item() if ml_timeStepequalHorizon_val.size == 1 else ml_timeStepequalHorizon_val[0,0] if ml_timeStepequalHorizon_val.size > 0 else None
                    else:
                        info['ml_timeStepequalHorizon'] = ml_timeStepequalHorizon_val
                
                ml_rtp_num_val = getattr(ml_rtp_final, 'num_generators', None)
                if ml_rtp_num_val is not None:
                    if isinstance(ml_rtp_num_val, np.ndarray):
                        info['ml_rtp_num'] = ml_rtp_num_val.item() if ml_rtp_num_val.size == 1 else ml_rtp_num_val[0,0] if ml_rtp_num_val.size > 0 else None
                    else:
                        info['ml_rtp_num'] = ml_rtp_num_val
        
        steps_info[step] = info

# Print results
print("\nStep | Python Path          | MATLAB Path          | Python Rtp | MATLAB Rtp")
print("-" * 80)
for step in sorted(steps_info.keys()):
    info = steps_info[step]
    py_path = "timeStepequalHorizon" if info.get('py_timeStepequalHorizon') else "normal" if info.get('py_timeStepequalHorizon') is False else "unknown"
    ml_path = "timeStepequalHorizon" if info.get('ml_timeStepequalHorizon') else "normal" if info.get('ml_timeStepequalHorizon') is False else "unknown"
    py_rtp = str(info.get('py_rtp_num', 'N/A'))
    ml_rtp = str(info.get('ml_rtp_num', 'N/A'))
    print(f"{step:4d} | {py_path:20s} | {ml_path:20s} | {py_rtp:10s} | {ml_rtp:10s}")

# Highlight Step 2
print("\n" + "=" * 80)
if 2 in steps_info:
    info = steps_info[2]
    print("Step 2 Details:")
    print(f"  Python timeStepequalHorizon: {info.get('py_timeStepequalHorizon')}")
    print(f"  MATLAB timeStepequalHorizon: {info.get('ml_timeStepequalHorizon')}")
    print(f"  Python Rtp generators: {info.get('py_rtp_num', 'N/A')}")
    print(f"  MATLAB Rtp generators: {info.get('ml_rtp_num', 'N/A')}")
    
    if info.get('py_timeStepequalHorizon') or info.get('ml_timeStepequalHorizon'):
        print("\n  → Step 2 uses timeStepequalHorizon path!")
        print("  → Need to compare Step 1's Rlintp (which becomes Rtp_h)")
else:
    print("Step 2: No tracking data found")
