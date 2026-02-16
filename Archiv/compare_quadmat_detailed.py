"""compare_quadmat_detailed - Detailed element-by-element comparison of quadMat"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("DETAILED quadMat COMPARISON: ELEMENT-BY-ELEMENT")
print("=" * 80)

# Load Python log
python_file = 'upstream_python_log.pkl'
if os.path.exists(python_file):
    with open(python_file, 'rb') as f:
        python_data = pickle.load(f)
    python_upstream = python_data.get('upstreamLog', [])
    print(f"\n[OK] Loaded Python log: {len(python_upstream)} entries")
else:
    print(f"\n[ERROR] Python log file not found: {python_file}")
    python_upstream = []

# Load MATLAB log
matlab_file = 'upstream_matlab_log.mat'
if os.path.exists(matlab_file):
    try:
        matlab_data = scipy.io.loadmat(matlab_file, squeeze_me=True)
        if 'upstreamLog' in matlab_data:
            matlab_upstream = matlab_data['upstreamLog']
        else:
            matlab_upstream = []
        print(f"[OK] Loaded MATLAB log: {len(matlab_upstream)} entries")
    except Exception as e:
        print(f"[ERROR] Could not load MATLAB log: {e}")
        matlab_upstream = []
else:
    print(f"[WARNING] MATLAB log file not found: {matlab_file}")
    matlab_upstream = []

# Helper function to extract value from MATLAB structured array
def get_ml_value(ml_obj, field, subfield=None):
    if hasattr(ml_obj, 'dtype') and field in ml_obj.dtype.names:
        val = ml_obj[field]
        if subfield and hasattr(val, 'dtype') and subfield in val.dtype.names:
            return val[subfield]
        return val
    return None

# Group Python entries by step and take the LAST entry for each
python_by_step = {}
for e in python_upstream:
    step = e.get('step', 0)
    if step not in python_by_step:
        python_by_step[step] = []
    python_by_step[step].append(e)

python_final_entries = {}
for step, entries in python_by_step.items():
    python_final_entries[step] = entries[-1]  # Last entry is converged

# Group MATLAB entries by step
matlab_by_step = {}
if len(matlab_upstream) > 0:
    for i in range(len(matlab_upstream)):
        e = matlab_upstream[i]
        if hasattr(e, 'dtype') and 'step' in e.dtype.names:
            step_val = e['step']
            step = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val
            matlab_by_step[int(step)] = i

# Find common steps with quadMat tracking
common_steps = sorted(set(python_final_entries.keys()) & set(matlab_by_step.keys()))

print(f"\nFound {len(common_steps)} common steps")
print(f"Comparing Steps 1-5 with quadMat tracking:\n")

for step in common_steps[:5]:
    py = python_final_entries[step]
    ml_idx = matlab_by_step[step]
    ml = matlab_upstream[ml_idx]
    
    py_quadmat = py.get('quadmat_tracking') if isinstance(py, dict) else None
    ml_quadmat = get_ml_value(ml, 'quadmat_tracking')
    
    if not py_quadmat or ml_quadmat is None:
        continue
    
    print(f"{'='*80}")
    print(f"Step {step}:")
    print(f"{'='*80}")
    
    # Compare for each dimension
    if isinstance(py_quadmat, list) and len(py_quadmat) > 0:
        for dim_idx, (dim, py_info) in enumerate(py_quadmat):
            print(f"\n  Dimension {dim}:")
            
            # Python quadMat
            py_diag = None
            py_full = None
            if 'dense_diag' in py_info:
                py_diag = np.asarray(py_info['dense_diag'])
            if 'dense_full' in py_info:
                py_full = np.asarray(py_info['dense_full'])
            
            # MATLAB quadMat
            ml_diag = None
            ml_full = None
            if isinstance(ml_quadmat, np.ndarray) and dim < len(ml_quadmat):
                ml_quadmat_dim = ml_quadmat[dim]
                if hasattr(ml_quadmat_dim, 'dtype') and ml_quadmat_dim.dtype.names:
                    if 'dense_diag' in ml_quadmat_dim.dtype.names:
                        ml_diag_val = ml_quadmat_dim['dense_diag']
                        # Extract from object array if needed
                        if isinstance(ml_diag_val, np.ndarray):
                            if ml_diag_val.dtype == object:
                                ml_diag_val = ml_diag_val.item()
                            if isinstance(ml_diag_val, np.ndarray):
                                ml_diag = ml_diag_val.flatten()
                            else:
                                ml_diag = np.asarray(ml_diag_val).flatten()
                    if 'dense_full' in ml_quadmat_dim.dtype.names:
                        ml_full_val = ml_quadmat_dim['dense_full']
                        # Extract from object array if needed
                        if isinstance(ml_full_val, np.ndarray):
                            if ml_full_val.dtype == object:
                                ml_full_val = ml_full_val.item()
                            if isinstance(ml_full_val, np.ndarray):
                                ml_full = ml_full_val
                            else:
                                ml_full = np.asarray(ml_full_val)
                            # Handle potential shape issues
                            if ml_full is not None:
                                if ml_full.ndim == 0 or ml_full.size == 0:
                                    ml_full = None
                elif isinstance(ml_quadmat_dim, dict):
                    # Handle as dict if not structured array
                    if 'dense_diag' in ml_quadmat_dim:
                        ml_diag_val = ml_quadmat_dim['dense_diag']
                        if isinstance(ml_diag_val, np.ndarray):
                            ml_diag = ml_diag_val.flatten()
                    if 'dense_full' in ml_quadmat_dim:
                        ml_full_val = ml_quadmat_dim['dense_full']
                        if isinstance(ml_full_val, np.ndarray):
                            ml_full = ml_full_val
            
            # Compare diagonal
            if py_diag is not None and ml_diag is not None:
                if len(py_diag) == len(ml_diag):
                    diag_diff = np.abs(py_diag - ml_diag)
                    diag_rel = diag_diff / (np.abs(ml_diag) + 1e-10) * 100
                    print(f"    Diagonal comparison:")
                    print(f"      Python: {py_diag}")
                    print(f"      MATLAB: {ml_diag}")
                    print(f"      Absolute difference: {diag_diff}")
                    print(f"      Relative difference: {diag_rel}%")
                    max_rel = np.max(diag_rel)
                    print(f"      Max relative difference: {max_rel:.6f}%")
                    if max_rel > 0.1:
                        print(f"      [WARNING] Significant difference!")
            
            # Compare full matrix
            if py_full is not None and ml_full is not None:
                if py_full.shape == ml_full.shape:
                    full_diff = np.abs(py_full - ml_full)
                    full_rel = full_diff / (np.abs(ml_full) + 1e-10) * 100
                    max_rel_full = np.max(full_rel)
                    print(f"    Full matrix comparison:")
                    print(f"      Shape: {py_full.shape}")
                    print(f"      Max absolute difference: {np.max(full_diff):.6e}")
                    print(f"      Max relative difference: {max_rel_full:.6f}%")
                    if max_rel_full > 0.1:
                        print(f"      [WARNING] Significant difference in full matrix!")
                        # Show where differences are largest
                        max_idx = np.unravel_index(np.argmax(full_rel), full_rel.shape)
                        print(f"      Largest difference at index {max_idx}:")
                        print(f"        Python: {py_full[max_idx]:.10e}")
                        print(f"        MATLAB: {ml_full[max_idx]:.10e}")
                        print(f"        Difference: {full_diff[max_idx]:.10e} ({full_rel[max_idx]:.6f}%)")
                else:
                    print(f"    Full matrix shape mismatch: Python {py_full.shape} vs MATLAB {ml_full.shape}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
