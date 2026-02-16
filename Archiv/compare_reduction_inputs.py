"""compare_reduction_inputs - Compare R and redFactor before reduction"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("COMPARING REDUCTION INPUTS AND OUTPUTS")
print("=" * 80)

# Load Python log
python_file = 'upstream_python_log.pkl'
if os.path.exists(python_file):
    with open(python_file, 'rb') as f:
        python_data = pickle.load(f)
    python_upstream = python_data.get('upstreamLog', [])
else:
    print(f"ERROR: {python_file} not found")
    python_upstream = []

# Load MATLAB log
matlab_file = 'upstream_matlab_log.mat'
if os.path.exists(matlab_file):
    matlab_data = scipy.io.loadmat(matlab_file, squeeze_me=False, struct_as_record=False)
    matlab_upstream = matlab_data.get('upstreamLog', [])
else:
    print(f"ERROR: {matlab_file} not found")
    matlab_upstream = []

# Helper function to get MATLAB struct field value
def get_ml_value(ml_obj, field):
    """Get field value from MATLAB struct, handling nested structures"""
    if hasattr(ml_obj, 'dtype') and ml_obj.dtype.names and field in ml_obj.dtype.names:
        # Structured array
        val = ml_obj[field]
        if isinstance(val, np.ndarray) and val.size > 0:
            if val.dtype == object and val.size == 1:
                val = val.item()
            elif hasattr(val, 'dtype') and val.dtype == object:
                val = val.item() if val.size == 1 else val[0,0]
        return val
    elif hasattr(ml_obj, field):
        # mat_struct object
        val = getattr(ml_obj, field)
        if isinstance(val, np.ndarray) and val.size > 0:
            # If it's a nested struct array, extract it
            if val.dtype == object or (hasattr(val, 'dtype') and val.dtype == object):
                val = val.item() if val.size == 1 else val[0,0]
            elif val.size == 1:
                val = val.item()
        return val
    return None

# Helper function to get nested MATLAB struct field
def get_ml_nested_value(ml_obj, field, nested_field):
    """Get nested field from MATLAB struct (e.g., Rmax_before_reduction.num_generators)"""
    parent = get_ml_value(ml_obj, field)
    if parent is None:
        return None
    # parent should be a mat_struct or dict-like object
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

# Group Python entries by step
python_by_step = {}
for e in python_upstream:
    step = e.get('step', 0)
    if step not in python_by_step:
        python_by_step[step] = []
    python_by_step[step].append(e)

# Find entries with R tracking (they have both R_before_reduction and Rred_after_reduction)
python_final_entries = {}
for step, entries in python_by_step.items():
    # Look for the last entry that has R tracking
    for entry in reversed(entries):
        if 'R_before_reduction' in entry and 'Rred_after_reduction' in entry:
            python_final_entries[step] = entry
            break
    # If no entry with R tracking found, use the last one
    if step not in python_final_entries:
        python_final_entries[step] = entries[-1]

# Group MATLAB entries by step
matlab_by_step = {}
# MATLAB log is a 2D array (n, 1)
for i in range(matlab_upstream.shape[0]):
    e = matlab_upstream[i, 0]  # Access as 2D array
    step_val = get_ml_value(e, 'step')
    if step_val is not None:
        if isinstance(step_val, np.ndarray):
            step = step_val.item() if step_val.size == 1 else step_val[0,0]
        else:
            step = step_val
        matlab_by_step[int(step)] = i

# Compare Step 3
step = 3
if step in python_final_entries and step in matlab_by_step:
    py = python_final_entries[step]
    ml_idx = matlab_by_step[step]
    # Access MATLAB entry correctly
    if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
        ml = matlab_upstream[ml_idx, 0]  # Access as 2D array
    else:
        ml = matlab_upstream[ml_idx]  # Access as 1D array
    
    print(f"\nStep {step} - Reduction Comparison:\n")
    
    # Compare Rmax, Rlinti, RallError before reduction
    py_Rmax = py.get('Rmax_before_reduction')
    ml_Rmax = get_ml_value(ml, 'Rmax_before_reduction')
    
    py_Rlinti = py.get('Rlinti_before_Rmax')
    ml_Rlinti = get_ml_value(ml, 'Rlinti_before_Rmax')
    
    py_RallError = py.get('RallError_before_Rmax')
    ml_RallError = get_ml_value(ml, 'RallError_before_Rmax')
    
    # Compare Rmax
    if py_Rmax and ml_Rmax:
        print("Rmax before reduction:")
        py_num_gen = py_Rmax.get('num_generators', 0) if isinstance(py_Rmax, dict) else 0
        ml_num_gen = get_ml_nested_value(ml, 'Rmax_before_reduction', 'num_generators')
        if ml_num_gen is not None and isinstance(ml_num_gen, np.ndarray):
            ml_num_gen = ml_num_gen.item() if ml_num_gen.size == 1 else ml_num_gen[0,0] if ml_num_gen.size > 0 else None
        
        print(f"  Python: {py_num_gen} generators")
        print(f"  MATLAB: {ml_num_gen} generators")
        if py_num_gen != ml_num_gen:
            print(f"  *** MISMATCH: {py_num_gen} vs {ml_num_gen} ***")
        else:
            print(f"  Match!")
    else:
        print("Rmax before reduction: NOT TRACKED")
        if not py_Rmax:
            print("  Python: Missing")
        if not ml_Rmax:
            print("  MATLAB: Missing")
    
    # Compare Rlinti
    if py_Rlinti and ml_Rlinti:
        print("\nRlinti before Rmax:")
        py_num_gen = py_Rlinti.get('num_generators', 0) if isinstance(py_Rlinti, dict) else 0
        ml_num_gen = get_ml_nested_value(ml, 'Rlinti_before_Rmax', 'num_generators')
        if ml_num_gen is not None and isinstance(ml_num_gen, np.ndarray):
            ml_num_gen = ml_num_gen.item() if ml_num_gen.size == 1 else ml_num_gen[0,0] if ml_num_gen.size > 0 else None
        
        print(f"  Python: {py_num_gen} generators")
        print(f"  MATLAB: {ml_num_gen} generators")
        if py_num_gen != ml_num_gen:
            print(f"  *** MISMATCH: {py_num_gen} vs {ml_num_gen} ***")
        else:
            print(f"  Match!")
    else:
        print("\nRlinti before Rmax: NOT TRACKED")
        if not py_Rlinti:
            print("  Python: Missing")
        if not ml_Rlinti:
            print("  MATLAB: Missing")
    
    # Compare RallError
    if py_RallError and ml_RallError:
        print("\nRallError before Rmax:")
        py_num_gen = py_RallError.get('num_generators', 0) if isinstance(py_RallError, dict) else 0
        ml_num_gen = get_ml_nested_value(ml, 'RallError_before_Rmax', 'num_generators')
        if ml_num_gen is not None and isinstance(ml_num_gen, np.ndarray):
            ml_num_gen = ml_num_gen.item() if ml_num_gen.size == 1 else ml_num_gen[0,0] if ml_num_gen.size > 0 else None
        
        print(f"  Python: {py_num_gen} generators")
        print(f"  MATLAB: {ml_num_gen} generators")
        if py_num_gen != ml_num_gen:
            print(f"  *** MISMATCH: {py_num_gen} vs {ml_num_gen} ***")
        else:
            print(f"  Match!")
    else:
        print("\nRallError before Rmax: NOT TRACKED")
        if not py_RallError:
            print("  Python: Missing")
        if not ml_RallError:
            print("  MATLAB: Missing")
    
    # Compare Rred after reduction
    py_Rred_after = py.get('Rred_after_reduction')
    ml_Rred_after = get_ml_value(ml, 'Rred_after_reduction')
    
    if py_Rred_after and ml_Rred_after:
        print("\nRred after reduction:")
        py_num_gen = py_Rred_after.get('num_generators', 0)
        ml_num_gen = ml_Rred_after.get('num_generators', 0) if isinstance(ml_Rred_after, dict) else get_ml_value(ml_Rred_after, 'num_generators')
        if isinstance(ml_num_gen, np.ndarray):
            ml_num_gen = ml_num_gen.item() if ml_num_gen.size == 1 else ml_num_gen
        
        print(f"  Python: {py_num_gen} generators")
        print(f"  MATLAB: {ml_num_gen} generators")
        if py_num_gen != ml_num_gen:
            print(f"  *** MISMATCH: {py_num_gen} vs {ml_num_gen} ***")
            print(f"  This is the ROOT CAUSE of the 20% difference!")
        else:
            print(f"  Match!")
    else:
        print("\nRred after reduction: NOT TRACKED")
        if not py_Rred_after:
            print("  Python: Missing")
        if not ml_Rred_after:
            print("  MATLAB: Missing")

else:
    print(f"\nStep {step} not found in both logs")
    if step not in python_final_entries:
        print("  Python: Missing")
    if step not in matlab_by_step:
        print("  MATLAB: Missing")

print("\n" + "=" * 80)
