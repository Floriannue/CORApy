"""Compare Rlinti between Python and MATLAB"""
import numpy as np
import pickle
import scipy.io as sio
import os

print("=" * 80)
print("COMPARING Rlinti (Linearized Reachable Set)")
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
    matlab_data = sio.loadmat(matlab_file, squeeze_me=False, struct_as_record=False)
    matlab_upstream = matlab_data.get('upstreamLog', [])
else:
    print(f"ERROR: {matlab_file} not found")
    matlab_upstream = []

# Helper function to get MATLAB struct field value
def get_ml_value(ml_obj, field):
    """Get field value from MATLAB struct, handling nested structures"""
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
    """Get nested field from MATLAB struct"""
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

# Find Step 3 entries
step = 3

# Python - look for entry with Rlinti_before_Rmax or Rlinti_tracking
py_entry = None
for e in python_upstream:
    if e.get('step') == step:
        if 'Rlinti_before_Rmax' in e or 'Rlinti_tracking' in e:
            py_entry = e
            break

# MATLAB
ml_entry = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            if isinstance(step_val, np.ndarray):
                s = step_val.item() if step_val.size == 1 else step_val[0,0] if step_val.size > 0 else None
            else:
                s = step_val
            if s == step:
                ml_entry = e
                break

if py_entry and ml_entry:
    print(f"\nStep {step} - Rlinti Comparison:\n")
    
    # Compare Rlinti - use Rlinti_before_Rmax if available, otherwise Rlinti_tracking
    py_Rlinti = py_entry.get('Rlinti_before_Rmax') or py_entry.get('Rlinti_tracking')
    ml_Rlinti = get_ml_value(ml_entry, 'Rlinti_before_Rmax') or get_ml_value(ml_entry, 'Rlinti_tracking')
    
    if py_Rlinti and ml_Rlinti:
        print("Rlinti (from tracking):")
        py_num_gen = py_Rlinti.get('num_generators', 0) if isinstance(py_Rlinti, dict) else 0
        # Try Rlinti_before_Rmax first, then Rlinti_tracking
        ml_num_gen = get_ml_nested_value(ml_entry, 'Rlinti_before_Rmax', 'num_generators')
        if ml_num_gen is None:
            ml_num_gen = get_ml_nested_value(ml_entry, 'Rlinti_tracking', 'num_generators')
        if ml_num_gen is not None and isinstance(ml_num_gen, np.ndarray):
            ml_num_gen = ml_num_gen.item() if ml_num_gen.size == 1 else ml_num_gen[0,0] if ml_num_gen.size > 0 else None
        
        print(f"  Python: {py_num_gen} generators")
        print(f"  MATLAB: {ml_num_gen} generators")
        if py_num_gen != ml_num_gen:
            print(f"  *** MISMATCH: {py_num_gen} vs {ml_num_gen} ***")
        else:
            print(f"  Match!")
        
        # Compare centers
        py_center = py_Rlinti.get('center')
        ml_center = get_ml_nested_value(ml_entry, 'Rlinti_before_Rmax', 'center')
        if ml_center is None:
            ml_center = get_ml_nested_value(ml_entry, 'Rlinti_tracking', 'center')
        if py_center is not None and ml_center is not None:
            if isinstance(ml_center, np.ndarray):
                ml_center = ml_center.flatten()
            py_center = np.asarray(py_center).flatten()
            print(f"\n  Center comparison:")
            print(f"  Python: {py_center}")
            print(f"  MATLAB: {ml_center}")
            if np.allclose(py_center, ml_center, rtol=1e-10):
                print(f"  Match!")
            else:
                print(f"  *** MISMATCH ***")
                diff = np.abs(py_center - ml_center)
                print(f"  Max difference: {np.max(diff):.6e}")
        
        # Compare generators (first few)
        py_gens = py_Rlinti.get('generators')
        ml_gens = get_ml_nested_value(ml_entry, 'Rlinti_before_Rmax', 'generators')
        if ml_gens is None:
            ml_gens = get_ml_nested_value(ml_entry, 'Rlinti_tracking', 'generators')
        if py_gens is not None and ml_gens is not None:
            py_gens = np.asarray(py_gens)
            if isinstance(ml_gens, np.ndarray):
                ml_gens = ml_gens
            print(f"\n  Generators shape:")
            print(f"  Python: {py_gens.shape}")
            print(f"  MATLAB: {ml_gens.shape if hasattr(ml_gens, 'shape') else 'N/A'}")
            
            if py_gens.shape[1] != ml_gens.shape[1] if hasattr(ml_gens, 'shape') and len(ml_gens.shape) > 1 else False:
                print(f"  *** Generator count mismatch: {py_gens.shape[1]} vs {ml_gens.shape[1] if hasattr(ml_gens, 'shape') and len(ml_gens.shape) > 1 else 'N/A'} ***")
    else:
        print("Rlinti_tracking: NOT FOUND")
        if not py_Rlinti:
            print("  Python: Missing")
        if not ml_Rlinti:
            print("  MATLAB: Missing")
else:
    print(f"\nStep {step} not found in both logs")
    if not py_entry:
        print("  Python: Missing")
    if not ml_entry:
        print("  MATLAB: Missing")
