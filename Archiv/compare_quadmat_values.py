"""compare_quadmat_values - Compare quadMat values between Python and MATLAB"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("COMPARING quadMat VALUES")
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
            subval = val[subfield]
            if isinstance(subval, np.ndarray):
                if subval.size == 1:
                    try:
                        return float(subval.item())
                    except (ValueError, TypeError):
                        return None
                else:
                    return subval
            return subval
        return val
    return None

# Compare quadMat values for Steps 1-5
if python_upstream and len(matlab_upstream) > 0:
    print("\n" + "=" * 80)
    print("COMPARING quadMat VALUES")
    print("=" * 80)
    
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
    for i in range(len(matlab_upstream)):
        e = matlab_upstream[i]
        if hasattr(e, 'dtype') and 'step' in e.dtype.names:
            step_val = e['step']
            step = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val
            matlab_by_step[int(step)] = i
    
    # Find common steps
    common_steps = sorted(set(python_final_entries.keys()) & set(matlab_by_step.keys()))
    
    print(f"\nFound {len(common_steps)} common steps")
    print(f"Comparing Steps 1-5 to find quadMat differences:\n")
    
    for step in common_steps[:5]:
        py = python_final_entries[step]
        ml_idx = matlab_by_step[step]
        ml = matlab_upstream[ml_idx]
        
        print(f"{'='*80}")
        print(f"Step {step}:")
        print(f"{'='*80}")
        
        # Compare quadMat tracking
        py_quadmat = py.get('quadmat_tracking') if isinstance(py, dict) else None
        ml_quadmat_val = get_ml_value(ml, 'quadmat_tracking')
        
        if py_quadmat:
            print(f"quadMat tracking comparison:")
            
            # Python quadMat tracking
            if isinstance(py_quadmat, list) and len(py_quadmat) > 0:
                for dim_idx, (dim, info) in enumerate(py_quadmat):
                    print(f"\n  Dimension {dim}:")
                    print(f"    Python quadMat type: {info.get('type', 'unknown')}")
                    print(f"    Python is_interval: {info.get('is_interval', False)}")
                    print(f"    Python is_sparse: {info.get('is_sparse', False)}")
                    
                    if info.get('dense_diag') is not None:
                        py_diag = np.asarray(info['dense_diag'])
                        print(f"    Python dense_diag: {py_diag}")
                        print(f"    Python dense_max: {info.get('dense_max', 'N/A')}")
                    
                    if info.get('after_convert_diag') is not None:
                        py_after = np.asarray(info['after_convert_diag'])
                        print(f"    Python after_convert_diag: {py_after}")
                    
                    # MATLAB quadMat tracking
                    if ml_quadmat_val is not None:
                        if isinstance(ml_quadmat_val, np.ndarray) and ml_quadmat_val.size > 0:
                            if dim < len(ml_quadmat_val):
                                ml_quadmat_dim = ml_quadmat_val[dim]
                                if hasattr(ml_quadmat_dim, 'dtype') and 'dense_diag' in ml_quadmat_dim.dtype.names:
                                    ml_diag = ml_quadmat_dim['dense_diag']
                                    if isinstance(ml_diag, np.ndarray):
                                        ml_diag = ml_diag.flatten()
                                    
                                    print(f"    MATLAB dense_diag: {ml_diag}")
                                    
                                    if info.get('dense_diag') is not None and ml_diag is not None:
                                        if len(py_diag) == len(ml_diag):
                                            diag_diff = np.abs(py_diag - ml_diag)
                                            diag_rel = diag_diff / (np.abs(ml_diag) + 1e-10)
                                            print(f"    Diagonal difference: {diag_diff}")
                                            print(f"    Relative difference: {diag_rel * 100}%")
                                            max_rel = np.max(diag_rel) * 100
                                            print(f"    Max relative difference: {max_rel:.4f}%")
                                            if max_rel > 1.0:
                                                print(f"    [WARNING] Large difference in quadMat diagonal!")
                        else:
                            print(f"    MATLAB: No quadmat_tracking data available")
                    else:
                        print(f"    MATLAB: No quadmat_tracking data available")
            else:
                print(f"    Python: No quadMat tracking data")
        else:
            print(f"    No quadMat tracking data for this step")
        
        print()

print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print("If quadMat diagonal differs -> issue in matrix multiplication")
print("If quadMat matches but errorSec differs -> issue in quadMap extraction")
print("=" * 80)
