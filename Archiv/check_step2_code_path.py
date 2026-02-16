"""Check which code path Step 2 uses and compare Rtp components"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("CHECKING STEP 2's CODE PATH AND Rtp COMPONENTS")
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

# Find Step 2 entries
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
                if hasattr(e, 'dtype') and e.dtype.names and 'Rtp_final_tracking' in e.dtype.names:
                    ml_step2 = e
                    break

if py_step2 and ml_step2:
    print(f"\nStep {step2} - Code Path Analysis:\n")
    
    # Check which code path was used
    py_rtp_final = py_step2.get('Rtp_final_tracking')
    ml_rtp_final_field = get_ml_value(ml_step2, 'Rtp_final_tracking')
    if isinstance(ml_rtp_final_field, np.ndarray) and ml_rtp_final_field.size > 0:
        ml_rtp_final = ml_rtp_final_field.item() if ml_rtp_final_field.size == 1 else ml_rtp_final_field[0,0]
    else:
        ml_rtp_final = ml_rtp_final_field
    
    py_timeStepequalHorizon = py_rtp_final.get('timeStepequalHorizon_used', None) if isinstance(py_rtp_final, dict) else None
    ml_timeStepequalHorizon = None
    if ml_rtp_final is not None:
        ml_timeStepequalHorizon_val = getattr(ml_rtp_final, 'timeStepequalHorizon_used', None)
        if ml_timeStepequalHorizon_val is not None:
            if isinstance(ml_timeStepequalHorizon_val, np.ndarray):
                ml_timeStepequalHorizon = ml_timeStepequalHorizon_val.item() if ml_timeStepequalHorizon_val.size == 1 else ml_timeStepequalHorizon_val[0,0] if ml_timeStepequalHorizon_val.size > 0 else None
            else:
                ml_timeStepequalHorizon = ml_timeStepequalHorizon_val
    
    print("1. Code Path Used:")
    print(f"   Python timeStepequalHorizon: {py_timeStepequalHorizon}")
    print(f"   MATLAB timeStepequalHorizon: {ml_timeStepequalHorizon}")
    
    if py_timeStepequalHorizon:
        print("   → Python used timeStepequalHorizon path (Rtp = Rtp_h + linx_h + Rerror_h)")
        # Check Rtp_h_tracking
        py_rtp_h = py_step2.get('Rtp_h_tracking')
        if py_rtp_h:
            py_rtp_h_num = py_rtp_h.get('num_generators', 0) if isinstance(py_rtp_h, dict) else 0
            print(f"   Rtp_h (from Step 1): {py_rtp_h_num} generators")
    else:
        print("   → Python used normal path (Rtp = Rlintp + nlnsys.linError.p.x + Rerror)")
        # Check Rlintp_tracking
        py_rlintp = py_step2.get('Rlintp_tracking')
        if py_rlintp:
            py_rlintp_num = py_rlintp.get('num_generators', 0) if isinstance(py_rlintp, dict) else 0
            print(f"   Rlintp: {py_rlintp_num} generators")
    
    if ml_timeStepequalHorizon:
        print("   → MATLAB used timeStepequalHorizon path (Rtp = Rtp_h + linx_h + Rerror_h)")
        # Check Rtp_h_tracking
        ml_rtp_h_field = get_ml_value(ml_step2, 'Rtp_h_tracking')
        if ml_rtp_h_field is not None:
            if isinstance(ml_rtp_h_field, np.ndarray) and ml_rtp_h_field.size > 0:
                ml_rtp_h = ml_rtp_h_field.item() if ml_rtp_h_field.size == 1 else ml_rtp_h_field[0,0]
            else:
                ml_rtp_h = ml_rtp_h_field
            if ml_rtp_h is not None:
                ml_rtp_h_num_val = getattr(ml_rtp_h, 'num_generators', None)
                if ml_rtp_h_num_val is not None:
                    if isinstance(ml_rtp_h_num_val, np.ndarray):
                        ml_rtp_h_num = ml_rtp_h_num_val.item() if ml_rtp_h_num_val.size == 1 else ml_rtp_h_num_val[0,0] if ml_rtp_h_num_val.size > 0 else None
                    else:
                        ml_rtp_h_num = ml_rtp_h_num_val
                    print(f"   Rtp_h (from Step 1): {ml_rtp_h_num} generators")
    else:
        print("   → MATLAB used normal path (Rtp = Rlintp + nlnsys.linError.p.x + Rerror)")
        # Check Rlintp_tracking
        ml_rlintp_field = get_ml_value(ml_step2, 'Rlintp_tracking')
        if ml_rlintp_field is not None:
            if isinstance(ml_rlintp_field, np.ndarray) and ml_rlintp_field.size > 0:
                ml_rlintp = ml_rlintp_field.item() if ml_rlintp_field.size == 1 else ml_rlintp_field[0,0]
            else:
                ml_rlintp = ml_rlintp_field
            if ml_rlintp is not None:
                ml_rlintp_num_val = getattr(ml_rlintp, 'num_generators', None)
                if ml_rlintp_num_val is not None:
                    if isinstance(ml_rlintp_num_val, np.ndarray):
                        ml_rlintp_num = ml_rlintp_num_val.item() if ml_rlintp_num_val.size == 1 else ml_rlintp_num_val[0,0] if ml_rlintp_num_val.size > 0 else None
                    else:
                        ml_rlintp_num = ml_rlintp_num_val
                    print(f"   Rlintp: {ml_rlintp_num} generators")
    
    # Compare Rerror
    print("\n2. Rerror Comparison:")
    py_rerror = py_step2.get('Rerror_tracking')
    ml_rerror_field = get_ml_value(ml_step2, 'Rerror_tracking')
    if isinstance(ml_rerror_field, np.ndarray) and ml_rerror_field.size > 0:
        ml_rerror = ml_rerror_field.item() if ml_rerror_field.size == 1 else ml_rerror_field[0,0]
    else:
        ml_rerror = ml_rerror_field
    
    if py_rerror and ml_rerror:
        py_rerror_num = py_rerror.get('num_generators', 0) if isinstance(py_rerror, dict) else 0
        ml_rerror_num = None
        if ml_rerror is not None:
            ml_rerror_num_val = getattr(ml_rerror, 'num_generators', None)
            if ml_rerror_num_val is not None:
                if isinstance(ml_rerror_num_val, np.ndarray):
                    ml_rerror_num = ml_rerror_num_val.item() if ml_rerror_num_val.size == 1 else ml_rerror_num_val[0,0] if ml_rerror_num_val.size > 0 else None
                else:
                    ml_rerror_num = ml_rerror_num_val
        
        print(f"   Python: {py_rerror_num} generators")
        print(f"   MATLAB: {ml_rerror_num} generators")
        if ml_rerror_num is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rerror_num != ml_rerror_num:
            print(f"   *** MISMATCH: {py_rerror_num} vs {ml_rerror_num} ***")
        else:
            print(f"   Match!")
    else:
        print("   Rerror_tracking not found")
    
    # Compare final Rtp
    print("\n3. Final Rtp Comparison:")
    py_rtp_num = py_rtp_final.get('num_generators', 0) if isinstance(py_rtp_final, dict) else 0
    ml_rtp_num = None
    if ml_rtp_final is not None:
        ml_rtp_num_val = getattr(ml_rtp_final, 'num_generators', None)
        if ml_rtp_num_val is not None:
            if isinstance(ml_rtp_num_val, np.ndarray):
                ml_rtp_num = ml_rtp_num_val.item() if ml_rtp_num_val.size == 1 else ml_rtp_num_val[0,0] if ml_rtp_num_val.size > 0 else None
            else:
                ml_rtp_num = ml_rtp_num_val
    
    print(f"   Python: {py_rtp_num} generators")
    print(f"   MATLAB: {ml_rtp_num} generators")
    if py_rtp_num != ml_rtp_num:
        print(f"   *** MISMATCH: {py_rtp_num} vs {ml_rtp_num} ***")
        print(f"   Difference: {abs(py_rtp_num - ml_rtp_num)} generators")
    else:
        print(f"   Match!")
else:
    print(f"Step {step2} entries not found")
    if not py_step2:
        print("  Python: Missing")
    if not ml_step2:
        print("  MATLAB: Missing")
