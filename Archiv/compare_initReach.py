"""Compare initReach_adaptive tracking between Python and MATLAB"""
import numpy as np
import pickle
import scipy.io as sio
import os

print("=" * 80)
print("COMPARING initReach_adaptive (Linearized Reachable Set Computation)")
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

# Python
py_entry = None
for e in python_upstream:
    if e.get('step') == step:
        if 'initReach_tracking' in e:
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
    print(f"\nStep {step} - initReach_adaptive Comparison:\n")
    
    py_track = py_entry.get('initReach_tracking')
    ml_track = get_ml_value(ml_entry, 'initReach_tracking')
    
    if py_track and ml_track:
        # Compare Rstart (input to initReach_adaptive)
        print("1. Rstart (input to initReach_adaptive):")
        py_rstart_gen = py_track.get('Rstart_num_generators', 0)
        ml_rstart_gen = get_ml_nested_value(ml_entry, 'initReach_tracking', 'Rstart_num_generators')
        if ml_rstart_gen is not None and isinstance(ml_rstart_gen, np.ndarray):
            ml_rstart_gen = ml_rstart_gen.item() if ml_rstart_gen.size == 1 else ml_rstart_gen[0,0] if ml_rstart_gen.size > 0 else None
        print(f"   Python: {py_rstart_gen} generators")
        print(f"   MATLAB: {ml_rstart_gen} generators")
        if py_rstart_gen != ml_rstart_gen:
            print(f"   *** MISMATCH: {py_rstart_gen} vs {ml_rstart_gen} ***")
        else:
            print(f"   Match!")
        
        # Compare Rhom (before reduction)
        print("\n2. Rhom (before reduction):")
        py_rhom_gen = py_track.get('Rhom_num_generators', 0)
        ml_rhom_gen = get_ml_nested_value(ml_entry, 'initReach_tracking', 'Rhom_num_generators')
        if ml_rhom_gen is not None and isinstance(ml_rhom_gen, np.ndarray):
            ml_rhom_gen = ml_rhom_gen.item() if ml_rhom_gen.size == 1 else ml_rhom_gen[0,0] if ml_rhom_gen.size > 0 else None
        print(f"   Python: {py_rhom_gen} generators")
        print(f"   MATLAB: {ml_rhom_gen} generators")
        if py_rhom_gen != ml_rhom_gen:
            print(f"   *** MISMATCH: {py_rhom_gen} vs {ml_rhom_gen} ***")
        else:
            print(f"   Match!")
        
        # Compare Rend.ti (after reduction - this is Rlinti)
        print("\n3. Rend.ti (after reduction - this becomes Rlinti):")
        py_rend_ti_gen = py_track.get('Rend_ti_num_generators', 0)
        ml_rend_ti_gen = get_ml_nested_value(ml_entry, 'initReach_tracking', 'Rend_ti_num_generators')
        if ml_rend_ti_gen is not None and isinstance(ml_rend_ti_gen, np.ndarray):
            ml_rend_ti_gen = ml_rend_ti_gen.item() if ml_rend_ti_gen.size == 1 else ml_rend_ti_gen[0,0] if ml_rend_ti_gen.size > 0 else None
        print(f"   Python: {py_rend_ti_gen} generators")
        print(f"   MATLAB: {ml_rend_ti_gen} generators")
        if py_rend_ti_gen != ml_rend_ti_gen:
            print(f"   *** MISMATCH: {py_rend_ti_gen} vs {ml_rend_ti_gen} ***")
            print(f"   This is the source of the Rlinti divergence!")
        else:
            print(f"   Match!")
        
        # Compare redFactor
        print("\n4. redFactor (used for reduction):")
        py_redFactor = py_track.get('redFactor')
        ml_redFactor = get_ml_nested_value(ml_entry, 'initReach_tracking', 'redFactor')
        if ml_redFactor is not None and isinstance(ml_redFactor, np.ndarray):
            ml_redFactor = ml_redFactor.item() if ml_redFactor.size == 1 else ml_redFactor[0,0] if ml_redFactor.size > 0 else None
        print(f"   Python: {py_redFactor}")
        print(f"   MATLAB: {ml_redFactor}")
        if py_redFactor is not None and ml_redFactor is not None:
            if abs(py_redFactor - ml_redFactor) > 1e-10:
                print(f"   *** MISMATCH: {py_redFactor} vs {ml_redFactor} ***")
            else:
                print(f"   Match!")
        
        # Compare intermediate values
        print("\n5. Intermediate values:")
        # Rhom_tp
        py_rhom_tp_gen = py_track.get('Rhom_tp_num_generators', 0)
        ml_rhom_tp_gen = get_ml_nested_value(ml_entry, 'initReach_tracking', 'Rhom_tp_num_generators')
        if ml_rhom_tp_gen is not None and isinstance(ml_rhom_tp_gen, np.ndarray):
            ml_rhom_tp_gen = ml_rhom_tp_gen.item() if ml_rhom_tp_gen.size == 1 else ml_rhom_tp_gen[0,0] if ml_rhom_tp_gen.size > 0 else None
        print(f"   Rhom_tp: Python {py_rhom_tp_gen} gens, MATLAB {ml_rhom_tp_gen} gens")
        if py_rhom_tp_gen != ml_rhom_tp_gen:
            print(f"   *** MISMATCH ***")
        
        # Rtrans
        py_rtrans_gen = py_track.get('Rtrans_num_generators', 0)
        ml_rtrans_gen = get_ml_nested_value(ml_entry, 'initReach_tracking', 'Rtrans_num_generators')
        if ml_rtrans_gen is not None and isinstance(ml_rtrans_gen, np.ndarray):
            ml_rtrans_gen = ml_rtrans_gen.item() if ml_rtrans_gen.size == 1 else ml_rtrans_gen[0,0] if ml_rtrans_gen.size > 0 else None
        print(f"   Rtrans: Python {py_rtrans_gen} gens, MATLAB {ml_rtrans_gen} gens")
        if py_rtrans_gen != ml_rtrans_gen:
            print(f"   *** MISMATCH ***")
        
        # inputCorr
        py_inputCorr_gen = py_track.get('inputCorr_num_generators', 0)
        ml_inputCorr_gen = get_ml_nested_value(ml_entry, 'initReach_tracking', 'inputCorr_num_generators')
        if ml_inputCorr_gen is not None and isinstance(ml_inputCorr_gen, np.ndarray):
            ml_inputCorr_gen = ml_inputCorr_gen.item() if ml_inputCorr_gen.size == 1 else ml_inputCorr_gen[0,0] if ml_inputCorr_gen.size > 0 else None
        print(f"   inputCorr: Python {py_inputCorr_gen} gens, MATLAB {ml_inputCorr_gen} gens")
        if py_inputCorr_gen != ml_inputCorr_gen:
            print(f"   *** MISMATCH ***")
    else:
        print("initReach_tracking: NOT FOUND")
        if not py_track:
            print("  Python: Missing")
        if not ml_track:
            print("  MATLAB: Missing")
else:
    print(f"\nStep {step} not found in both logs")
    if not py_entry:
        print("  Python: Missing")
    if not ml_entry:
        print("  MATLAB: Missing")
