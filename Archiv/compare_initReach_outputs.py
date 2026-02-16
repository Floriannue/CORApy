"""Compare Step 3's initReach_adaptive outputs: Rend.ti and Rend.tp"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("COMPARING STEP 3's initReach_adaptive OUTPUTS")
print("=" * 80)
print("Rend.ti and Rend.tp - these become Rlinti and Rlintp")
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

# Find Step 3 entries - try to find any entry with step 3
step3 = 3
py_step3 = None
for e in python_upstream:
    if e.get('step') == step3:
        # Check if it has initReach_tracking or if we can find it in a later entry
        if 'initReach_tracking' in e:
            py_step3 = e
            break
        # If not, store the first step 3 entry we find
        if py_step3 is None:
            py_step3 = e

# Also check if initReach_tracking is in a separate entry
if py_step3 and 'initReach_tracking' not in py_step3:
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
                # Check if it has initReach_tracking
                if hasattr(e, 'dtype') and e.dtype.names and 'initReach_tracking' in e.dtype.names:
                    ml_step3 = e
                    break
                # If not, store the first step 3 entry we find
                if ml_step3 is None:
                    ml_step3 = e

if py_step3 and ml_step3:
    print(f"\nStep {step3} - initReach_adaptive outputs:\n")
    
    py_init = py_step3.get('initReach_tracking')
    if hasattr(ml_step3, 'dtype') and ml_step3.dtype.names and 'initReach_tracking' in ml_step3.dtype.names:
        ml_init_field = ml_step3['initReach_tracking']
        if isinstance(ml_init_field, np.ndarray) and ml_init_field.size > 0:
            ml_init = ml_init_field.item() if ml_init_field.size == 1 else ml_init_field[0,0]
        else:
            ml_init = ml_init_field
    else:
        ml_init = None
    
    if py_init and ml_init:
        # Compare Rend.ti
        py_rend_ti = py_init.get('Rend_ti_num_generators', 0)
        ml_rend_ti_val = getattr(ml_init, 'Rend_ti_num_generators', None) if ml_init else None
        if ml_rend_ti_val is not None:
            if isinstance(ml_rend_ti_val, np.ndarray):
                ml_rend_ti = ml_rend_ti_val.item() if ml_rend_ti_val.size == 1 else ml_rend_ti_val[0,0] if ml_rend_ti_val.size > 0 else None
            else:
                ml_rend_ti = ml_rend_ti_val
        else:
            ml_rend_ti = None
        
        print("1. Rend.ti (becomes Rlinti):")
        print(f"   Python: {py_rend_ti} generators")
        print(f"   MATLAB: {ml_rend_ti} generators")
        if ml_rend_ti is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rend_ti != ml_rend_ti:
            print(f"   *** MISMATCH: {py_rend_ti} vs {ml_rend_ti} ***")
            print(f"   Difference: {abs(py_rend_ti - ml_rend_ti)} generators")
        else:
            print(f"   Match!")
        
        # Compare Rend.tp
        py_rend_tp = py_init.get('Rend_tp_num_generators', 0)
        ml_rend_tp_val = getattr(ml_init, 'Rend_tp_num_generators', None) if ml_init else None
        if ml_rend_tp_val is not None:
            if isinstance(ml_rend_tp_val, np.ndarray):
                ml_rend_tp = ml_rend_tp_val.item() if ml_rend_tp_val.size == 1 else ml_rend_tp_val[0,0] if ml_rend_tp_val.size > 0 else None
            else:
                ml_rend_tp = ml_rend_tp_val
        else:
            ml_rend_tp = None
        
        print("\n2. Rend.tp (becomes Rlintp):")
        print(f"   Python: {py_rend_tp} generators")
        print(f"   MATLAB: {ml_rend_tp} generators")
        if ml_rend_tp is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rend_tp != ml_rend_tp:
            print(f"   *** MISMATCH: {py_rend_tp} vs {ml_rend_tp} ***")
            print(f"   Difference: {abs(py_rend_tp - ml_rend_tp)} generators")
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
        
        print("\n3. Rhom_tp (input to Rend.tp reduction):")
        print(f"   Python: {py_rhom_tp} generators")
        print(f"   MATLAB: {ml_rhom_tp} generators")
        if ml_rhom_tp is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rhom_tp != ml_rhom_tp:
            print(f"   *** MISMATCH: {py_rhom_tp} vs {ml_rhom_tp} ***")
            print(f"   Difference: {abs(py_rhom_tp - ml_rhom_tp)} generators")
            print(f"   This is the input to Rend.tp reduction!")
        else:
            print(f"   Match!")
        
        # Compare Rhom (input to Rend.ti reduction)
        py_rhom = py_init.get('Rhom_num_generators', 0)
        ml_rhom_val = getattr(ml_init, 'Rhom_num_generators', None) if ml_init else None
        if ml_rhom_val is not None:
            if isinstance(ml_rhom_val, np.ndarray):
                ml_rhom = ml_rhom_val.item() if ml_rhom_val.size == 1 else ml_rhom_val[0,0] if ml_rhom_val.size > 0 else None
            else:
                ml_rhom = ml_rhom_val
        else:
            ml_rhom = None
        
        print("\n4. Rhom (input to Rend.ti reduction):")
        print(f"   Python: {py_rhom} generators")
        print(f"   MATLAB: {ml_rhom} generators")
        if ml_rhom is None:
            print(f"   *** MATLAB tracking not accessible ***")
        elif py_rhom != ml_rhom:
            print(f"   *** MISMATCH: {py_rhom} vs {ml_rhom} ***")
            print(f"   Difference: {abs(py_rhom - ml_rhom)} generators")
            print(f"   This is the input to Rend.ti reduction!")
        else:
            print(f"   Match!")
        
        print("\n" + "=" * 80)
        print("ANALYSIS:")
        print("=" * 80)
        if py_rend_ti == 2 and ml_rend_ti == 7:
            print("*** FOUND: Rend.ti has 2 vs 7 generators difference! ***")
            print("This is the 'init reach 2 vs 7 generators difference' you asked about.")
            print("Rend.ti becomes Rlinti, which is used in Rmax = Rlinti + RallError")
            print("This explains part of the divergence chain!")
        elif py_rend_tp == 2 and ml_rend_tp == 7:
            print("*** FOUND: Rend.tp has 2 vs 7 generators difference! ***")
            print("This is the 'init reach 2 vs 7 generators difference' you asked about.")
        else:
            print("The 2 vs 7 difference might be in Rend.ti or Rend.tp")
            print("Check the values above to see which one matches.")
    else:
        print("initReach_tracking not found")
else:
    print(f"Step {step3} entries not found")
