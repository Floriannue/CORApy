"""Trace Step 2's Rtp components to find the 14 vs 16 generators difference"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("TRACING STEP 2's Rtp COMPONENTS")
print("=" * 80)
print("Rtp = Rlintp + nlnsys.linError.p.x + Rerror")
print("Step 2's Rtp BEFORE reduction: Python 14, MATLAB 16 (difference: 2)")
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

print("\n1. Rlintp (from Rend.tp in initReach_adaptive):")
print("-" * 80)

# Find initReach_tracking for Step 2
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
                if hasattr(e, 'dtype') and e.dtype.names and 'initReach_tracking' in e.dtype.names:
                    ml_step2_init = e
                    break

if py_step2_init and ml_step2_init:
    py_init = py_step2_init.get('initReach_tracking')
    if hasattr(ml_step2_init, 'dtype') and ml_step2_init.dtype.names and 'initReach_tracking' in ml_step2_init.dtype.names:
        ml_init_field = ml_step2_init['initReach_tracking']
        if isinstance(ml_init_field, np.ndarray) and ml_init_field.size > 0:
            ml_init = ml_init_field.item() if ml_init_field.size == 1 else ml_init_field[0,0]
        else:
            ml_init = ml_init_field
    else:
        ml_init = None
    
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
        
        print(f"   Rend.tp (becomes Rlintp):")
        print(f"   Python: {py_rend_tp} generators")
        print(f"   MATLAB: {ml_rend_tp} generators")
        if ml_rend_tp is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rend_tp != ml_rend_tp:
            print(f"   *** MISMATCH: {py_rend_tp} vs {ml_rend_tp} ***")
            print(f"   Difference: {abs(py_rend_tp - ml_rend_tp)} generators")
        else:
            print(f"   Match!")
    else:
        print("   initReach_tracking not found")
else:
    print("   Step 2 initReach_tracking not found")

# Check Rlintp_tracking directly
print("\n2. Rlintp_tracking (from linReach_adaptive):")
print("-" * 80)

py_step2_rlintp = None
for e in python_upstream:
    if e.get('step') == step2 and 'Rlintp_tracking' in e:
        py_step2_rlintp = e
        break

ml_step2_rlintp = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step2:
                if hasattr(e, 'dtype') and e.dtype.names and 'Rlintp_tracking' in e.dtype.names:
                    ml_step2_rlintp = e
                    break

if py_step2_rlintp and ml_step2_rlintp:
    py_rlintp = py_step2_rlintp.get('Rlintp_tracking')
    ml_rlintp_field = get_ml_value(ml_step2_rlintp, 'Rlintp_tracking')
    if isinstance(ml_rlintp_field, np.ndarray) and ml_rlintp_field.size > 0:
        ml_rlintp = ml_rlintp_field.item() if ml_rlintp_field.size == 1 else ml_rlintp_field[0,0]
    else:
        ml_rlintp = ml_rlintp_field
    
    if py_rlintp and ml_rlintp:
        py_rlintp_num = py_rlintp.get('num_generators', 0) if isinstance(py_rlintp, dict) else 0
        ml_rlintp_num = None
        if ml_rlintp is not None:
            ml_rlintp_num_val = getattr(ml_rlintp, 'num_generators', None)
            if ml_rlintp_num_val is not None:
                if isinstance(ml_rlintp_num_val, np.ndarray):
                    ml_rlintp_num = ml_rlintp_num_val.item() if ml_rlintp_num_val.size == 1 else ml_rlintp_num_val[0,0] if ml_rlintp_num_val.size > 0 else None
                else:
                    ml_rlintp_num = ml_rlintp_num_val
        
        print(f"   Rlintp (before adding Rerror):")
        print(f"   Python: {py_rlintp_num} generators")
        print(f"   MATLAB: {ml_rlintp_num} generators")
        if ml_rlintp_num is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rlintp_num != ml_rlintp_num:
            print(f"   *** MISMATCH: {py_rlintp_num} vs {ml_rlintp_num} ***")
            print(f"   Difference: {abs(py_rlintp_num - ml_rlintp_num)} generators")
        else:
            print(f"   Match!")
    else:
        print("   Rlintp_tracking not found")
else:
    print("   Step 2 Rlintp_tracking not found")

# Check Rerror_tracking
print("\n3. Rerror_tracking (from errorSolution_adaptive):")
print("-" * 80)

py_step2_rerror = None
for e in python_upstream:
    if e.get('step') == step2 and 'Rerror_tracking' in e:
        py_step2_rerror = e
        break

ml_step2_rerror = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step2:
                if hasattr(e, 'dtype') and e.dtype.names and 'Rerror_tracking' in e.dtype.names:
                    ml_step2_rerror = e
                    break

if py_step2_rerror and ml_step2_rerror:
    py_rerror = py_step2_rerror.get('Rerror_tracking')
    ml_rerror_field = get_ml_value(ml_step2_rerror, 'Rerror_tracking')
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
        
        print(f"   Rerror (before adding to Rtp):")
        print(f"   Python: {py_rerror_num} generators")
        print(f"   MATLAB: {ml_rerror_num} generators")
        if ml_rerror_num is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rerror_num != ml_rerror_num:
            print(f"   *** MISMATCH: {py_rerror_num} vs {ml_rerror_num} ***")
            print(f"   Difference: {abs(py_rerror_num - ml_rerror_num)} generators")
        else:
            print(f"   Match!")
    else:
        print("   Rerror_tracking not found")
else:
    print("   Step 2 Rerror_tracking not found")

# Compare with actual Rtp before reduction
print("\n4. Actual Rtp BEFORE reduction in reach_adaptive:")
print("-" * 80)

python_Rtp_tracking = python_data.get('Rtp_tracking', {})
py_step2_rtp = python_Rtp_tracking.get(step2 + 1, {})  # Step 2's Rtp becomes Step 3's Rstart
py_before = py_step2_rtp.get('before', {})

matlab_Rtp_tracking = matlab_data.get('Rtp_tracking', None)
ml_step2_rtp = None
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

if py_before and ml_step2_rtp:
    py_num = py_before.get('num_generators', 0)
    ml_before_field = getattr(ml_step2_rtp, 'before', None) if ml_step2_rtp else None
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
    
    print(f"   Rtp before reduction:")
    print(f"   Python: {py_num} generators")
    print(f"   MATLAB: {ml_num} generators")
    if py_num != ml_num:
        print(f"   *** MISMATCH: {py_num} vs {ml_num} ***")
        print(f"   Difference: {abs(py_num - ml_num)} generators")
    else:
        print(f"   Match!")
    
    # Try to compute expected: Rlintp + Rerror
    print("\n5. Expected Rtp = Rlintp + Rerror:")
    print("-" * 80)
    if py_step2_rlintp and py_step2_rerror:
        py_rlintp_num = py_step2_rlintp.get('Rlintp_tracking', {}).get('num_generators', 0) if isinstance(py_step2_rlintp.get('Rlintp_tracking'), dict) else 0
        py_rerror_num = py_step2_rerror.get('Rerror_tracking', {}).get('num_generators', 0) if isinstance(py_step2_rerror.get('Rerror_tracking'), dict) else 0
        py_expected = py_rlintp_num + py_rerror_num
        print(f"   Python: {py_rlintp_num} + {py_rerror_num} = {py_expected} generators")
        print(f"   Actual: {py_num} generators")
        if py_expected == py_num:
            print(f"   ✅ Expected matches actual!")
        else:
            print(f"   ⚠️  Expected {py_expected} but got {py_num} (difference: {py_num - py_expected})")
    
    if ml_step2_rlintp and ml_step2_rerror:
        ml_rlintp_num = None
        ml_rerror_num = None
        ml_rlintp_field = get_ml_value(ml_step2_rlintp, 'Rlintp_tracking')
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
        
        ml_rerror_field = get_ml_value(ml_step2_rerror, 'Rerror_tracking')
        if ml_rerror_field is not None:
            if isinstance(ml_rerror_field, np.ndarray) and ml_rerror_field.size > 0:
                ml_rerror = ml_rerror_field.item() if ml_rerror_field.size == 1 else ml_rerror_field[0,0]
            else:
                ml_rerror = ml_rerror_field
            if ml_rerror is not None:
                ml_rerror_num_val = getattr(ml_rerror, 'num_generators', None)
                if ml_rerror_num_val is not None:
                    if isinstance(ml_rerror_num_val, np.ndarray):
                        ml_rerror_num = ml_rerror_num_val.item() if ml_rerror_num_val.size == 1 else ml_rerror_num_val[0,0] if ml_rerror_num_val.size > 0 else None
                    else:
                        ml_rerror_num = ml_rerror_num_val
        
        if ml_rlintp_num is not None and ml_rerror_num is not None:
            ml_expected = ml_rlintp_num + ml_rerror_num
            print(f"   MATLAB: {ml_rlintp_num} + {ml_rerror_num} = {ml_expected} generators")
            print(f"   Actual: {ml_num} generators")
            if ml_expected == ml_num:
                print(f"   ✅ Expected matches actual!")
            else:
                print(f"   ⚠️  Expected {ml_expected} but got {ml_num} (difference: {ml_num - ml_expected})")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print("If Rlintp differs, the issue is in initReach_adaptive (Rend.tp)")
print("If Rerror differs, the issue is in errorSolution_adaptive")
print("If both match but Rtp doesn't, the issue is in the addition operation")
