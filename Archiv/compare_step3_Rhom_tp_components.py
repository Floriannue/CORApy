"""Compare Step 3's Rhom_tp components: Rhom_tp = eAt * Rstart + Rtrans"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("COMPARING STEP 3's Rhom_tp COMPONENTS")
print("=" * 80)
print("Rhom_tp = eAt * Rstart + Rtrans")
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

# Find Step 3 entries
step3 = 3

# 1. Compare Rstart (input to initReach_adaptive)
print("\n1. Rstart (input to initReach_adaptive):")
py_step3_rstart = None
for e in python_upstream:
    if e.get('step') == step3 and 'Rstart_tracking' in e:
        py_step3_rstart = e
        break

ml_step3_rstart = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step3:
                if hasattr(e, 'dtype') and e.dtype.names and 'Rstart_tracking' in e.dtype.names:
                    ml_step3_rstart = e
                    break

if py_step3_rstart and ml_step3_rstart:
    py_rstart = py_step3_rstart.get('Rstart_tracking')
    ml_rstart_field = get_ml_value(ml_step3_rstart, 'Rstart_tracking')
    if isinstance(ml_rstart_field, np.ndarray) and ml_rstart_field.size > 0:
        ml_rstart = ml_rstart_field.item() if ml_rstart_field.size == 1 else ml_rstart_field[0,0]
    else:
        ml_rstart = ml_rstart_field
    
    if py_rstart and ml_rstart:
        py_rstart_num = py_rstart.get('num_generators', 0) if isinstance(py_rstart, dict) else 0
        ml_rstart_num = None
        if ml_rstart is not None:
            ml_rstart_num_val = getattr(ml_rstart, 'num_generators', None)
            if ml_rstart_num_val is not None:
                if isinstance(ml_rstart_num_val, np.ndarray):
                    ml_rstart_num = ml_rstart_num_val.item() if ml_rstart_num_val.size == 1 else ml_rstart_num_val[0,0] if ml_rstart_num_val.size > 0 else None
                else:
                    ml_rstart_num = ml_rstart_num_val
        
        print(f"   Python: {py_rstart_num} generators")
        print(f"   MATLAB: {ml_rstart_num} generators")
        if ml_rstart_num is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rstart_num != ml_rstart_num:
            print(f"   *** MISMATCH: {py_rstart_num} vs {ml_rstart_num} ***")
            print(f"   Difference: {abs(py_rstart_num - ml_rstart_num)} generators")
        else:
            print(f"   Match!")

# 2. Compare Rtrans from initReach_tracking
print("\n2. Rtrans (from initReach_adaptive):")
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
        py_rtrans = py_init.get('Rtrans_num_generators', 0)
        ml_rtrans_val = getattr(ml_init, 'Rtrans_num_generators', None) if ml_init else None
        if ml_rtrans_val is not None:
            if isinstance(ml_rtrans_val, np.ndarray):
                ml_rtrans = ml_rtrans_val.item() if ml_rtrans_val.size == 1 else ml_rtrans_val[0,0] if ml_rtrans_val.size > 0 else None
            else:
                ml_rtrans = ml_rtrans_val
        else:
            ml_rtrans = None
        
        print(f"   Python: {py_rtrans} generators")
        print(f"   MATLAB: {ml_rtrans} generators")
        if ml_rtrans is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rtrans != ml_rtrans:
            print(f"   *** MISMATCH: {py_rtrans} vs {ml_rtrans} ***")
            print(f"   Difference: {abs(py_rtrans - ml_rtrans)} generators")
        else:
            print(f"   Match!")
        
        # Compare Rhom_tp
        py_rhom_tp = py_init.get('Rhom_tp_num_generators', 0)
        ml_rhom_tp_val = getattr(ml_init, 'Rhom_tp_num_generators', None) if ml_init else None
        if ml_rhom_tp_val is not None:
            if isinstance(ml_rhom_tp_val, np.ndarray):
                ml_rhom_tp = ml_rhom_tp_val.item() if ml_rhom_tp_val.size == 1 else ml_rhom_tp_val[0,0] if ml_rhom_tp_val.size > 0 else None
            else:
                ml_rhom_tp = ml_rhom_tp_val
        else:
            ml_rhom_tp = None
        
        print("\n3. Rhom_tp = eAt * Rstart + Rtrans:")
        print(f"   Python: {py_rhom_tp} generators")
        print(f"   MATLAB: {ml_rhom_tp} generators")
        if ml_rhom_tp is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rhom_tp != ml_rhom_tp:
            print(f"   *** MISMATCH: {py_rhom_tp} vs {ml_rhom_tp} ***")
            print(f"   Difference: {abs(py_rhom_tp - ml_rhom_tp)} generators")
            
            # Expected: Rhom_tp = Rstart + Rtrans (eAt multiplication doesn't change generator count)
            if py_rstart and ml_rstart and py_init and ml_init:
                py_expected = py_rstart_num + py_rtrans
                ml_expected = ml_rstart_num + ml_rtrans if ml_rstart_num is not None and ml_rtrans is not None else None
                print(f"\n   Expected: Rstart + Rtrans")
                print(f"   Python: {py_rstart_num} + {py_rtrans} = {py_expected}")
                if ml_expected is not None:
                    print(f"   MATLAB: {ml_rstart_num} + {ml_rtrans} = {ml_expected}")
                    if py_expected == py_rhom_tp and ml_expected == ml_rhom_tp:
                        print(f"   ✅ Expected matches actual!")
                    else:
                        print(f"   ⚠️  Python: Expected {py_expected} but got {py_rhom_tp}")
                        print(f"   ⚠️  MATLAB: Expected {ml_expected} but got {ml_rhom_tp}")
        else:
            print(f"   Match!")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("If Rstart differs, that's the root cause (we know it's 2 vs 4)")
print("If Rtrans differs, that could contribute to the difference")
print("If Rhom_tp doesn't match Rstart + Rtrans, there's an issue in the computation")
