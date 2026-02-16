"""Compare Step 3's Rend.tp from initReach_adaptive - this is the source of the 2-generator difference"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("COMPARING STEP 3's Rend.tp FROM initReach_adaptive")
print("=" * 80)
print("Rend.tp becomes Rlintp, which has 2 vs 4 generators difference")
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

# Find Step 3 entries with initReach_tracking
step3 = 3
py_step3 = None
for e in python_upstream:
    if e.get('step') == step3 and 'initReach_tracking' in e:
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
                if get_ml_value(e, 'initReach_tracking') is not None:
                    ml_step3 = e
                    break

if py_step3 and ml_step3:
    print(f"\nStep {step3} - Rend.tp from initReach_adaptive:\n")
    
    py_init = py_step3.get('initReach_tracking')
    # Access MATLAB struct field
    if hasattr(ml_step3, 'dtype') and ml_step3.dtype.names and 'initReach_tracking' in ml_step3.dtype.names:
        ml_init_field = ml_step3['initReach_tracking']
        if isinstance(ml_init_field, np.ndarray) and ml_init_field.size > 0:
            ml_init = ml_init_field.item() if ml_init_field.size == 1 else ml_init_field[0,0]
        else:
            ml_init = ml_init_field
    elif hasattr(ml_step3, 'initReach_tracking'):
        ml_init_field = getattr(ml_step3, 'initReach_tracking')
        if isinstance(ml_init_field, np.ndarray) and ml_init_field.size > 0:
            ml_init = ml_init_field.item() if ml_init_field.size == 1 else ml_init_field[0,0]
        else:
            ml_init = ml_init_field
    else:
        ml_init = None
    
    if py_init and ml_init:
        # Compare Rend.tp (this becomes Rlintp)
        py_rend_tp = py_init.get('Rend_tp_num_generators', 0)
        ml_rend_tp_val = getattr(ml_init, 'Rend_tp_num_generators', None) if ml_init else None
        if ml_rend_tp_val is not None:
            if isinstance(ml_rend_tp_val, np.ndarray):
                ml_rend_tp = ml_rend_tp_val.item() if ml_rend_tp_val.size == 1 else ml_rend_tp_val[0,0] if ml_rend_tp_val.size > 0 else None
            else:
                ml_rend_tp = ml_rend_tp_val
        else:
            ml_rend_tp = None
        
        print("Rend.tp (after reduction in initReach_adaptive):")
        print(f"   Python: {py_rend_tp} generators")
        print(f"   MATLAB: {ml_rend_tp} generators")
        if ml_rend_tp is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rend_tp != ml_rend_tp:
            print(f"   *** MISMATCH: {py_rend_tp} vs {ml_rend_tp} ***")
            print(f"   Difference: {abs(py_rend_tp - ml_rend_tp)} generators")
            print(f"   This is the source of the divergence!")
        else:
            print(f"   Match!")
        
        # Compare Rhom (before reduction) to see if input differs
        py_rhom = py_init.get('Rhom_num_generators', 0)
        ml_rhom_val = getattr(ml_init, 'Rhom_num_generators', None) if ml_init else None
        if ml_rhom_val is not None:
            if isinstance(ml_rhom_val, np.ndarray):
                ml_rhom = ml_rhom_val.item() if ml_rhom_val.size == 1 else ml_rhom_val[0,0] if ml_rhom_val.size > 0 else None
            else:
                ml_rhom = ml_rhom_val
        else:
            ml_rhom = None
        
        print("\nRhom (before reduction in initReach_adaptive):")
        print(f"   Python: {py_rhom} generators")
        print(f"   MATLAB: {ml_rhom} generators")
        if ml_rhom is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rhom != ml_rhom:
            print(f"   *** MISMATCH: {py_rhom} vs {ml_rhom} ***")
            print(f"   Difference: {abs(py_rhom - ml_rhom)} generators")
            print(f"   The input to reduction differs!")
        else:
            print(f"   Match!")
        
        # Compare redFactor used for reduction
        py_redFactor = py_init.get('redFactor')
        ml_redFactor_val = getattr(ml_init, 'redFactor', None) if ml_init else None
        if ml_redFactor_val is not None:
            if isinstance(ml_redFactor_val, np.ndarray):
                ml_redFactor = ml_redFactor_val.item() if ml_redFactor_val.size == 1 else ml_redFactor_val[0,0] if ml_redFactor_val.size > 0 else None
            else:
                ml_redFactor = ml_redFactor_val
        else:
            ml_redFactor = None
        
        print("\nredFactor used for reduction:")
        print(f"   Python: {py_redFactor}")
        print(f"   MATLAB: {ml_redFactor}")
        if py_redFactor is not None and ml_redFactor is not None:
            if abs(py_redFactor - ml_redFactor) > 1e-10:
                print(f"   *** MISMATCH: {py_redFactor} vs {ml_redFactor} ***")
            else:
                print(f"   Match!")
        
        # Compare Rhom_tp (input to Rend.tp reduction)
        py_rhom_tp = py_init.get('Rhom_tp_num_generators', 0)
        ml_rhom_tp_val = getattr(ml_init, 'Rhom_tp_num_generators', None) if ml_init else None
        if ml_rhom_tp_val is not None:
            if isinstance(ml_rhom_tp_val, np.ndarray):
                ml_rhom_tp = ml_rhom_tp_val.item() if ml_rhom_tp_val.size == 1 else ml_rhom_tp_val[0,0] if ml_rhom_tp_val.size > 0 else None
            else:
                ml_rhom_tp = ml_rhom_tp_val
        else:
            ml_rhom_tp = None
        
        print("\nRhom_tp (input to Rend.tp reduction):")
        print(f"   Python: {py_rhom_tp} generators")
        print(f"   MATLAB: {ml_rhom_tp} generators")
        if ml_rhom_tp is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rhom_tp != ml_rhom_tp:
            print(f"   *** MISMATCH: {py_rhom_tp} vs {ml_rhom_tp} ***")
            print(f"   Difference: {abs(py_rhom_tp - ml_rhom_tp)} generators")
        else:
            print(f"   Match!")
    else:
        print("initReach_tracking not found")
else:
    print(f"Step {step3} entries not found")
