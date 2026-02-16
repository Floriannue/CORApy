"""Check Step 2's Rtp_final_tracking to see the final Rtp after adding Rerror"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("CHECKING STEP 2's Rtp_final_tracking")
print("=" * 80)
print("This is Rtp = Rlintp + nlnsys.linError.p.x + Rerror")
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
                if hasattr(e, 'dtype') and e.dtype.names and 'Rtp_final_tracking' in e.dtype.names:
                    ml_step2 = e
                    break

if py_step2 and ml_step2:
    print(f"\nStep {step2} - Rtp_final_tracking (final Rtp from linReach_adaptive):\n")
    
    py_rtp_final = py_step2.get('Rtp_final_tracking')
    ml_rtp_final_field = get_ml_value(ml_step2, 'Rtp_final_tracking')
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
        
        print("Rtp_final_tracking (after adding Rerror):")
        print(f"   Python: {py_num} generators")
        print(f"   MATLAB: {ml_num} generators")
        if ml_num is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_num != ml_num:
            print(f"   *** MISMATCH: {py_num} vs {ml_num} ***")
            print(f"   Difference: {abs(py_num - ml_num)} generators")
        else:
            print(f"   Match!")
        
        # Compare with Rtp before reduction in reach_adaptive
        print("\nComparing with Rtp before reduction in reach_adaptive:")
        python_Rtp_tracking = python_data.get('Rtp_tracking', {})
        py_step2_rtp = python_Rtp_tracking.get(step2 + 1, {})
        py_before = py_step2_rtp.get('before', {})
        py_before_num = py_before.get('num_generators', 0)
        
        matlab_Rtp_tracking = matlab_data.get('Rtp_tracking', None)
        ml_before_num = None
        if matlab_Rtp_tracking is not None:
            if isinstance(matlab_Rtp_tracking, np.ndarray) and matlab_Rtp_tracking.size > 0:
                rtp_obj = matlab_Rtp_tracking.item() if matlab_Rtp_tracking.size == 1 else matlab_Rtp_tracking[0,0]
                step_field = f'step_{step2 + 1}'
                if hasattr(rtp_obj, step_field):
                    ml_step2_field = getattr(rtp_obj, step_field)
                    if isinstance(ml_step2_field, np.ndarray) and ml_step2_field.size > 0:
                        ml_step2_rtp = ml_step2_field.item() if ml_step2_field.size == 1 else ml_step2_field[0,0]
                    else:
                        ml_step2_rtp = ml_step2_field
                    ml_before_field = getattr(ml_step2_rtp, 'before', None) if ml_step2_rtp else None
                    if isinstance(ml_before_field, np.ndarray) and ml_before_field.size > 0:
                        ml_before = ml_before_field.item() if ml_before_field.size == 1 else ml_before_field[0,0]
                    else:
                        ml_before = ml_before_field
                    if ml_before is not None:
                        ml_before_num_val = getattr(ml_before, 'num_generators', None)
                        if ml_before_num_val is not None:
                            if isinstance(ml_before_num_val, np.ndarray):
                                ml_before_num = ml_before_num_val.item() if ml_before_num_val.size == 1 else ml_before_num_val[0,0] if ml_before_num_val.size > 0 else None
                            else:
                                ml_before_num = ml_before_num_val
        
        print(f"   Rtp_final_tracking: Python {py_num}, MATLAB {ml_num}")
        print(f"   Rtp before reduction: Python {py_before_num}, MATLAB {ml_before_num}")
        
        if py_num == py_before_num and (ml_num is None or ml_num == ml_before_num):
            print(f"   ✅ Rtp_final_tracking matches Rtp before reduction!")
        else:
            print(f"   ⚠️  Rtp_final_tracking differs from Rtp before reduction")
            if py_num != py_before_num:
                print(f"      Python: {py_num} vs {py_before_num} (difference: {py_before_num - py_num})")
            if ml_num is not None and ml_num != ml_before_num:
                print(f"      MATLAB: {ml_num} vs {ml_before_num} (difference: {ml_before_num - ml_num})")
    else:
        print("Rtp_final_tracking not found")
else:
    print(f"Step {step2} entries not found")
    if not py_step2:
        print("  Python: Missing")
    if not ml_step2:
        print("  MATLAB: Missing")
