"""Compare Rtp before and after reduction in reach_adaptive for Step 2"""
import numpy as np
import pickle
import scipy.io as sio

print("=" * 80)
print("COMPARING Rtp IN reach_adaptive (Step 2)")
print("=" * 80)

# Load logs
python_file = 'upstream_python_log.pkl'
with open(python_file, 'rb') as f:
    python_data = pickle.load(f)
python_Rtp_tracking = python_data.get('Rtp_tracking', {})

matlab_file = 'upstream_matlab_log.mat'
matlab_data = sio.loadmat(matlab_file, squeeze_me=False, struct_as_record=False)
matlab_Rtp_tracking = matlab_data.get('Rtp_tracking', None)

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

# Compare Step 2
step2 = 2

print(f"\n=== STEP {step2} - Rtp in reach_adaptive ===\n")

# Python - Rtp_tracking is keyed by the step number that will USE this Rtp (i.e., step 2's Rtp is used by step 3)
# So step 2's Rtp is stored under key 3
py_step2 = python_Rtp_tracking.get(step2 + 1, {})  # Step 2's Rtp becomes Step 3's Rstart
py_before = py_step2.get('before', {})
py_after = py_step2.get('after', {})

# MATLAB - Rtp_tracking is keyed by the step number that will USE this Rtp
# So step 2's Rtp is stored under step_3
ml_step2 = None
if matlab_Rtp_tracking is not None:
    # MATLAB stores as struct with fields like step_2, step_3, etc.
    step_field = f'step_{step2 + 1}'  # Step 2's Rtp becomes Step 3's Rstart
    # Extract the struct object from numpy array
    if isinstance(matlab_Rtp_tracking, np.ndarray) and matlab_Rtp_tracking.size > 0:
        rtp_obj = matlab_Rtp_tracking.item() if matlab_Rtp_tracking.size == 1 else matlab_Rtp_tracking[0,0]
        # Access the field directly
        if hasattr(rtp_obj, step_field):
            ml_step2_field = getattr(rtp_obj, step_field)
            # ml_step2_field might be a numpy array containing the struct
            if isinstance(ml_step2_field, np.ndarray) and ml_step2_field.size > 0:
                ml_step2 = ml_step2_field.item() if ml_step2_field.size == 1 else ml_step2_field[0,0]
            else:
                ml_step2 = ml_step2_field

if py_before and ml_step2:
    # Compare before reduction
    print("1. Rtp BEFORE reduction in reach_adaptive:")
    py_num_before = py_before.get('num_generators', 0)
    # Access MATLAB struct fields
    ml_before_field = getattr(ml_step2, 'before', None) if ml_step2 else None
    # ml_before_field might be a numpy array
    if isinstance(ml_before_field, np.ndarray) and ml_before_field.size > 0:
        ml_before = ml_before_field.item() if ml_before_field.size == 1 else ml_before_field[0,0]
    else:
        ml_before = ml_before_field
    
    ml_num_before = None
    if ml_before is not None:
        ml_num_before_val = getattr(ml_before, 'num_generators', None)
        if ml_num_before_val is not None:
            if isinstance(ml_num_before_val, np.ndarray):
                ml_num_before = ml_num_before_val.item() if ml_num_before_val.size == 1 else ml_num_before_val[0,0] if ml_num_before_val.size > 0 else None
            else:
                ml_num_before = ml_num_before_val
    
    print(f"   Python: {py_num_before} generators")
    print(f"   MATLAB: {ml_num_before} generators")
    if py_num_before != ml_num_before:
        print(f"   *** MISMATCH: {py_num_before} vs {ml_num_before} ***")
        print(f"   This is the source of the divergence!")
    else:
        print(f"   Match!")
    
    # Compare redFactor
    py_redFactor = py_before.get('redFactor')
    ml_redFactor = None
    if ml_before is not None:
        ml_redFactor_val = getattr(ml_before, 'redFactor', None)
        if ml_redFactor_val is not None:
            if isinstance(ml_redFactor_val, np.ndarray):
                ml_redFactor = ml_redFactor_val.item() if ml_redFactor_val.size == 1 else ml_redFactor_val[0,0] if ml_redFactor_val.size > 0 else None
            else:
                ml_redFactor = ml_redFactor_val
    print(f"\n   redFactor used for reduction:")
    print(f"   Python: {py_redFactor}")
    print(f"   MATLAB: {ml_redFactor}")
    if py_redFactor is not None and ml_redFactor is not None:
        if abs(py_redFactor - ml_redFactor) > 1e-10:
            print(f"   *** MISMATCH: {py_redFactor} vs {ml_redFactor} ***")
        else:
            print(f"   Match!")
    
    # Compare after reduction
    ml_after_field = getattr(ml_step2, 'after', None) if ml_step2 else None
    if isinstance(ml_after_field, np.ndarray) and ml_after_field.size > 0:
        ml_after = ml_after_field.item() if ml_after_field.size == 1 else ml_after_field[0,0]
    else:
        ml_after = ml_after_field
    
    if py_after and ml_after:
        print(f"\n2. Rtp AFTER reduction in reach_adaptive:")
        py_num_after = py_after.get('num_generators', 0)
        ml_num_after = None
        if ml_after is not None:
            ml_num_after_val = getattr(ml_after, 'num_generators', None)
            if ml_num_after_val is not None:
                if isinstance(ml_num_after_val, np.ndarray):
                    ml_num_after = ml_num_after_val.item() if ml_num_after_val.size == 1 else ml_num_after_val[0,0] if ml_num_after_val.size > 0 else None
                else:
                    ml_num_after = ml_num_after_val
        
        print(f"   Python: {py_num_after} generators")
        print(f"   MATLAB: {ml_num_after} generators")
        if py_num_after != ml_num_after:
            print(f"   *** MISMATCH: {py_num_after} vs {ml_num_after} ***")
            print(f"   This becomes Step 3's Rstart!")
        else:
            print(f"   Match!")
    else:
        print("\n2. Rtp AFTER reduction: NOT TRACKED")
else:
    print("Step 2 Rtp tracking not found")
    if not py_before:
        print("  Python: Missing")
    if not ml_step2:
        print("  MATLAB: Missing")
