"""compare_hessian_values - Compare H (Hessian) values between Python and MATLAB"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("COMPARING H (HESSIAN) VALUES")
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

# Compare H values for Steps 1-5
if python_upstream and len(matlab_upstream) > 0:
    print("\n" + "=" * 80)
    print("COMPARING H (HESSIAN) VALUES")
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
    print(f"Comparing Steps 1-5 to find H differences:\n")
    
    for step in common_steps[:5]:
        py = python_final_entries[step]
        ml_idx = matlab_by_step[step]
        ml = matlab_upstream[ml_idx]
        
        print(f"{'='*80}")
        print(f"Step {step}:")
        print(f"{'='*80}")
        
        # Compare H before quadMap
        py_h = py.get('H_before_quadmap') if isinstance(py, dict) else None
        ml_h_val = get_ml_value(ml, 'H_before_quadmap')
        
        if py_h and ml_h_val is not None:
            # H is a list/array of matrices (one per dimension)
            py_h_list = py_h if isinstance(py_h, (list, tuple)) else [py_h]
            
            # MATLAB H is stored as a cell array
            if hasattr(ml_h_val, 'dtype') and ml_h_val.dtype.names:
                # Structured array - might be cell array
                ml_h_list = []
                for i in range(len(ml_h_val)):
                    if hasattr(ml_h_val[i], 'dtype') and 'max_abs' in ml_h_val[i].dtype.names:
                        ml_h_list.append(ml_h_val[i])
            else:
                # Try to access as array
                try:
                    ml_h_list = [ml_h_val[i] for i in range(len(ml_h_val))] if hasattr(ml_h_val, '__len__') else [ml_h_val]
                except:
                    ml_h_list = [ml_h_val]
            
            print(f"H (Hessian) comparison:")
            print(f"  Python has {len(py_h_list)} H matrices")
            print(f"  MATLAB has {len(ml_h_list)} H matrices")
            
            # Compare max_abs for each dimension
            for dim in range(min(len(py_h_list), len(ml_h_list))):
                py_h_dim = py_h_list[dim]
                ml_h_dim = ml_h_list[dim] if dim < len(ml_h_list) else None
                
                if py_h_dim and ml_h_dim is not None:
                    py_max_abs = py_h_dim.get('max_abs') if isinstance(py_h_dim, dict) else None
                    ml_max_abs = get_ml_value(ml_h_dim, 'max_abs') if hasattr(ml_h_dim, 'dtype') else None
                    
                    if py_max_abs is not None and ml_max_abs is not None:
                        h_diff = abs(py_max_abs - ml_max_abs)
                        h_rel = h_diff / max(abs(ml_max_abs), 1e-10)
                        print(f"  Dimension {dim}:")
                        print(f"    Python max_abs: {py_max_abs:.6e}")
                        print(f"    MATLAB max_abs: {ml_max_abs:.6e}")
                        print(f"    Difference: {h_diff:.6e} ({h_rel*100:.4f}%)")
                        if h_rel > 0.1:
                            print(f"    [WARNING] Large difference in H[{dim}]!")
        
        print()

print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print("If H differs -> issue in Hessian computation")
print("If H matches but errorSec differs -> issue in quadMap computation")
print("=" * 80)
