"""Check the 2 vs 7 generators difference in Rlinti (from Rend.ti)"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("CHECKING 2 vs 7 GENERATORS DIFFERENCE IN Rlinti")
print("=" * 80)
print("Rlinti comes from Rend.ti in initReach_adaptive")
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

# Find Step 3 entries with Rlinti_before_Rmax
step3 = 3
py_step3 = None
for e in python_upstream:
    if e.get('step') == step3 and 'Rlinti_before_Rmax' in e:
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
                if hasattr(e, 'dtype') and e.dtype.names and 'Rlinti_before_Rmax' in e.dtype.names:
                    ml_step3 = e
                    break

if py_step3 and ml_step3:
    print(f"\nStep {step3} - Rlinti (from Rend.ti):\n")
    
    py_rlinti = py_step3.get('Rlinti_before_Rmax')
    ml_rlinti_field = get_ml_value(ml_step3, 'Rlinti_before_Rmax')
    if isinstance(ml_rlinti_field, np.ndarray) and ml_rlinti_field.size > 0:
        ml_rlinti = ml_rlinti_field.item() if ml_rlinti_field.size == 1 else ml_rlinti_field[0,0]
    else:
        ml_rlinti = ml_rlinti_field
    
    if py_rlinti and ml_rlinti:
        py_rlinti_num = py_rlinti.get('num_generators', 0) if isinstance(py_rlinti, dict) else 0
        ml_rlinti_num = None
        if ml_rlinti is not None:
            ml_rlinti_num_val = getattr(ml_rlinti, 'num_generators', None)
            if ml_rlinti_num_val is not None:
                if isinstance(ml_rlinti_num_val, np.ndarray):
                    ml_rlinti_num = ml_rlinti_num_val.item() if ml_rlinti_num_val.size == 1 else ml_rlinti_num_val[0,0] if ml_rlinti_num_val.size > 0 else None
                else:
                    ml_rlinti_num = ml_rlinti_num_val
        
        print("Rlinti (becomes part of Rmax = Rlinti + RallError):")
        print(f"   Python: {py_rlinti_num} generators")
        print(f"   MATLAB: {ml_rlinti_num} generators")
        if ml_rlinti_num is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rlinti_num != ml_rlinti_num:
            print(f"   *** MISMATCH: {py_rlinti_num} vs {ml_rlinti_num} ***")
            print(f"   Difference: {abs(py_rlinti_num - ml_rlinti_num)} generators")
            if py_rlinti_num == 2 and ml_rlinti_num == 7:
                print(f"\n   *** THIS IS THE 2 vs 7 GENERATORS DIFFERENCE! ***")
                print(f"   Rend.ti (from initReach_adaptive) becomes Rlinti")
                print(f"   Python reduces to 2 generators, MATLAB reduces to 7 generators")
                print(f"   This difference propagates to Rmax = Rlinti + RallError")
        else:
            print(f"   Match!")
    else:
        print("Rlinti_before_Rmax not found")
else:
    print(f"Step {step3} entries not found")

# Also check if we can find Rend.ti from initReach_tracking
print("\n" + "=" * 80)
print("ALSO CHECKING Rend.ti FROM initReach_tracking:")
print("=" * 80)

py_step3_init = None
for e in python_upstream:
    if e.get('step') == step3 and 'initReach_tracking' in e:
        py_step3_init = e
        break

ml_step3_init = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step3:
                if hasattr(e, 'dtype') and e.dtype.names and 'initReach_tracking' in e.dtype.names:
                    ml_step3_init = e
                    break

if py_step3_init and ml_step3_init:
    py_init = py_step3_init.get('initReach_tracking')
    if hasattr(ml_step3_init, 'dtype') and ml_step3_init.dtype.names and 'initReach_tracking' in ml_step3_init.dtype.names:
        ml_init_field = ml_step3_init['initReach_tracking']
        if isinstance(ml_init_field, np.ndarray) and ml_init_field.size > 0:
            ml_init = ml_init_field.item() if ml_init_field.size == 1 else ml_init_field[0,0]
        else:
            ml_init = ml_init_field
    else:
        ml_init = None
    
    if py_init and ml_init:
        py_rend_ti = py_init.get('Rend_ti_num_generators', 0)
        ml_rend_ti_val = getattr(ml_init, 'Rend_ti_num_generators', None) if ml_init else None
        if ml_rend_ti_val is not None:
            if isinstance(ml_rend_ti_val, np.ndarray):
                ml_rend_ti = ml_rend_ti_val.item() if ml_rend_ti_val.size == 1 else ml_rend_ti_val[0,0] if ml_rend_ti_val.size > 0 else None
            else:
                ml_rend_ti = ml_rend_ti_val
        else:
            ml_rend_ti = None
        
        print("\nRend.ti (from initReach_adaptive, becomes Rlinti):")
        print(f"   Python: {py_rend_ti} generators")
        print(f"   MATLAB: {ml_rend_ti} generators")
        if ml_rend_ti is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rend_ti != ml_rend_ti:
            print(f"   *** MISMATCH: {py_rend_ti} vs {ml_rend_ti} ***")
            print(f"   Difference: {abs(py_rend_ti - ml_rend_ti)} generators")
            if py_rend_ti == 2 and ml_rend_ti == 7:
                print(f"\n   *** THIS IS THE 2 vs 7 GENERATORS DIFFERENCE! ***")
                print(f"   Rend.ti is reduced differently in Python vs MATLAB")
        else:
            print(f"   Match!")
