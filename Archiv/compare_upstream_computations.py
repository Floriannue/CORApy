"""compare_upstream_computations - Compare upstream computations between Python and MATLAB"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("COMPARING UPSTREAM COMPUTATIONS: PYTHON vs MATLAB")
print("=" * 80)

# Load Python log
python_file = 'upstream_python_log.pkl'
if os.path.exists(python_file):
    with open(python_file, 'rb') as f:
        python_data = pickle.load(f)
    python_upstream = python_data.get('upstreamLog', [])
    python_optimaldeltat = python_data.get('optimaldeltatLog', [])
    print(f"\n[OK] Loaded Python log: {len(python_upstream)} upstream entries, {len(python_optimaldeltat)} optimaldeltat entries")
else:
    print(f"\n[ERROR] Python log file not found: {python_file}")
    print("Please run track_upstream_python.py first")
    python_upstream = []
    python_optimaldeltat = []

# Load MATLAB log
matlab_file = 'upstream_matlab_log.mat'
if os.path.exists(matlab_file):
    try:
        matlab_data = scipy.io.loadmat(matlab_file, squeeze_me=True)
        if 'upstreamLog' in matlab_data:
            matlab_log_struct = matlab_data['upstreamLog']
            # Keep as structured array - don't convert to dicts
            if isinstance(matlab_log_struct, np.ndarray):
                matlab_upstream = matlab_log_struct
            else:
                matlab_upstream = []
        else:
            matlab_upstream = []
        print(f"[OK] Loaded MATLAB log: {len(matlab_upstream)} upstream entries")
    except Exception as e:
        import traceback
        print(f"[ERROR] Could not load MATLAB log: {e}")
        traceback.print_exc()
        matlab_upstream = []
else:
    print(f"[WARNING] MATLAB log file not found: {matlab_file}")
    print("Please run track_upstream_matlab.m first")
    matlab_upstream = []

# Compare entries
if python_upstream and len(matlab_upstream) > 0:
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    # Filter entries that have Rerror_before_optimaldeltat (these are the ones used in _aux_optimaldeltat)
    python_rerror_entries = [e for e in python_upstream if e.get('Rerror_before_optimaldeltat') and e['Rerror_before_optimaldeltat'].get('rerr1') is not None]
    
    # MATLAB entries are structured arrays (numpy.void), access fields directly
    # Use same logic as inspect_matlab_log.py which works
    matlab_rerror_entries = []
    for i in range(len(matlab_upstream)):
        e = matlab_upstream[i]
        if hasattr(e, 'dtype') and 'Rerror_before_optimaldeltat' in e.dtype.names:
            re = e['Rerror_before_optimaldeltat']
            if re is not None and hasattr(re, 'size') and re.size > 0:
                if hasattr(re, 'dtype') and 'rerr1' in re.dtype.names:
                    rerr1_val = re['rerr1']
                    if isinstance(rerr1_val, np.ndarray) and rerr1_val.size == 1:
                        try:
                            val = float(rerr1_val.item())
                            if not np.isnan(val) and not np.isinf(val):
                                matlab_rerror_entries.append(i)  # Store index instead of entry
                        except (ValueError, TypeError):
                            pass
    
    print(f"\nFound {len(python_rerror_entries)} Python entries with Rerror")
    print(f"Found {len(matlab_rerror_entries)} MATLAB entries with Rerror")
    
    # Group Python entries by (step, run) and take the LAST entry for each (converged iteration)
    python_by_step_run = {}
    for e in python_rerror_entries:
        step = e.get('step', 0)
        run = e.get('run', 0)
        key = (step, run)
        # Keep the last entry for each (step, run) pair (converged iteration)
        if key not in python_by_step_run:
            python_by_step_run[key] = []
        python_by_step_run[key].append(e)
    
    # For each (step, run), take the last entry
    python_final_entries = {}
    for key, entries in python_by_step_run.items():
        python_final_entries[key] = entries[-1]  # Last entry is the converged one
    
    # Group MATLAB entries by (step, run)
    matlab_by_step_run = {}
    for i in matlab_rerror_entries:
        e = matlab_upstream[i]
        if hasattr(e, 'dtype') and 'step' in e.dtype.names and 'run' in e.dtype.names:
            step_val = e['step']
            step = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val
            run_val = e['run']
            run = run_val.item() if isinstance(run_val, np.ndarray) and run_val.size == 1 else run_val
            key = (int(step), int(run))
            matlab_by_step_run[key] = i
    
    # Find common (step, run) pairs
    common_keys = set(python_final_entries.keys()) & set(matlab_by_step_run.keys())
    common_keys_sorted = sorted(common_keys, key=lambda x: (x[0], x[1]))  # Sort by step, then run
    
    print(f"\nFound {len(common_keys_sorted)} common (step, run) pairs")
    print(f"Comparing first {min(20, len(common_keys_sorted))} converged iterations:\n")
    
    num_compare = min(20, len(common_keys_sorted))
    
    for i in range(num_compare):
        key = common_keys_sorted[i]
        step, run = key
        py = python_final_entries[key]
        ml_idx = matlab_by_step_run[key]
        ml = matlab_upstream[ml_idx]
        
        # Extract step numbers
        py_step = py.get('step', 0) if isinstance(py, dict) else (py['step'].item() if hasattr(py, 'dtype') and 'step' in py.dtype.names else 0)
        
        # Extract MATLAB step (handle int or array)
        if hasattr(ml, 'dtype') and 'step' in ml.dtype.names:
            step_val = ml['step']
            ml_step = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val
        else:
            ml_step = 0
        
        print(f"Step {step}, Run {run} (converged iteration):")
        
        # Helper function to extract value from MATLAB structured array or dict
        def get_ml_value(ml_obj, field, subfield=None):
            if hasattr(ml_obj, 'dtype') and field in ml_obj.dtype.names:
                val = ml_obj[field]
                if subfield and hasattr(val, 'dtype') and subfield in val.dtype.names:
                    subval = val[subfield]
                    # Handle object arrays
                    if isinstance(subval, np.ndarray):
                        if subval.size == 1:
                            try:
                                return float(subval.item())
                            except (ValueError, TypeError):
                                return None
                        else:
                            return subval
                    else:
                        return float(subval) if subval is not None else None
                return val
            elif isinstance(ml_obj, dict):
                val = ml_obj.get(field)
                if subfield and isinstance(val, dict):
                    return val.get(subfield)
                return val
            return None
        
        # Compare VerrorDyn before errorSolution
        py_vd = py.get('VerrorDyn_before_errorsolution') if isinstance(py, dict) else None
        ml_vd_val = get_ml_value(ml, 'VerrorDyn_before_errorsolution')
        if py_vd and ml_vd_val is not None:
            py_radius_max = py_vd.get('radius_max') if isinstance(py_vd, dict) else None
            ml_radius_max = get_ml_value(ml_vd_val, 'radius_max') if ml_vd_val is not None else None
            if py_radius_max is not None and ml_radius_max is not None:
                vd_diff = abs(py_radius_max - ml_radius_max)
                vd_rel = vd_diff / max(abs(ml_radius_max), 1e-10)
                print(f"  VerrorDyn before errorSolution:")
                print(f"    Python radius_max: {py_radius_max:.6e}")
                print(f"    MATLAB radius_max: {ml_radius_max:.6e}")
                print(f"    Difference: {vd_diff:.6e} ({vd_rel*100:.4f}%)")
        
        # Compare Rerror before optimaldeltat
        py_re = py.get('Rerror_before_optimaldeltat') if isinstance(py, dict) else None
        ml_re_val = get_ml_value(ml, 'Rerror_before_optimaldeltat')
        if py_re and ml_re_val is not None:
            py_rerr1 = py_re.get('rerr1') if isinstance(py_re, dict) else None
            ml_rerr1 = get_ml_value(ml_re_val, 'rerr1') if ml_re_val is not None else None
            if py_rerr1 is not None and ml_rerr1 is not None:
                rerr1_diff = abs(py_rerr1 - ml_rerr1)
                rerr1_rel = rerr1_diff / max(abs(ml_rerr1), 1e-10)
                print(f"  Rerror before optimaldeltat:")
                print(f"    Python rerr1: {py_rerr1:.6e}")
                print(f"    MATLAB rerr1: {ml_rerr1:.6e}")
                print(f"    Difference: {rerr1_diff:.6e} ({rerr1_rel*100:.4f}%)")
                if rerr1_rel > 0.1:
                    print(f"    [WARNING] Large difference in rerr1!")
        
        # Compare errorSec if available (this comes from quadMap)
        py_es = py.get('errorSec_before_combine') if isinstance(py, dict) else None
        ml_es_val = get_ml_value(ml, 'errorSec_before_combine')
        if py_es and ml_es_val is not None:
            py_radius_max = py_es.get('radius_max') if isinstance(py_es, dict) else None
            ml_radius_max = get_ml_value(ml_es_val, 'radius_max') if ml_es_val is not None else None
            if py_radius_max is not None and ml_radius_max is not None:
                es_diff = abs(py_radius_max - ml_radius_max)
                es_rel = es_diff / max(abs(ml_radius_max), 1e-10)
                print(f"  errorSec before combine (from quadMap):")
                print(f"    Python radius_max: {py_radius_max:.6e}")
                print(f"    MATLAB radius_max: {ml_radius_max:.6e}")
                print(f"    Difference: {es_diff:.6e} ({es_rel*100:.4f}%)")
                if es_rel > 0.1:
                    print(f"    [WARNING] Large difference in errorSec!")
        
        # Compare errorLagr if available
        py_el = py.get('errorLagr_before_combine') if isinstance(py, dict) else None
        ml_el_val = get_ml_value(ml, 'errorLagr_before_combine')
        if py_el and ml_el_val is not None:
            py_radius_max = py_el.get('radius_max') if isinstance(py_el, dict) else None
            ml_radius_max = get_ml_value(ml_el_val, 'radius_max') if ml_el_val is not None else None
            if py_radius_max is not None and ml_radius_max is not None:
                el_diff = abs(py_radius_max - ml_radius_max)
                el_rel = el_diff / max(abs(ml_radius_max), 1e-10)
                print(f"  errorLagr before combine:")
                print(f"    Python radius_max: {py_radius_max:.6e}")
                print(f"    MATLAB radius_max: {ml_radius_max:.6e}")
                print(f"    Difference: {el_diff:.6e} ({el_rel*100:.4f}%)")
        
        # Compare VerrorDyn before reduce (errorSec + errorLagr)
        py_vd = py.get('VerrorDyn_before_reduce') if isinstance(py, dict) else None
        ml_vd_val = get_ml_value(ml, 'VerrorDyn_before_reduce')
        if py_vd and ml_vd_val is not None:
            py_radius_max = py_vd.get('radius_max') if isinstance(py_vd, dict) else None
            ml_radius_max = get_ml_value(ml_vd_val, 'radius_max') if ml_vd_val is not None else None
            if py_radius_max is not None and ml_radius_max is not None:
                vd_diff = abs(py_radius_max - ml_radius_max)
                vd_rel = vd_diff / max(abs(ml_radius_max), 1e-10)
                print(f"  VerrorDyn before reduce (errorSec + errorLagr):")
                print(f"    Python radius_max: {py_radius_max:.6e}")
                print(f"    MATLAB radius_max: {ml_radius_max:.6e}")
                print(f"    Difference: {vd_diff:.6e} ({vd_rel*100:.4f}%)")
                if vd_rel > 0.1:
                    print(f"    [WARNING] Large difference in VerrorDyn before reduce!")
        
        # Compare Z before quadMap (input to quadMap)
        py_z = py.get('Z_before_quadmap') if isinstance(py, dict) else None
        ml_z_val = get_ml_value(ml, 'Z_before_quadmap')
        if py_z and ml_z_val is not None:
            py_radius_max = py_z.get('radius_max') if isinstance(py_z, dict) else None
            ml_radius_max = get_ml_value(ml_z_val, 'radius_max') if ml_z_val is not None else None
            if py_radius_max is not None and ml_radius_max is not None:
                z_diff = abs(py_radius_max - ml_radius_max)
                z_rel = z_diff / max(abs(ml_radius_max), 1e-10)
                print(f"  Z before quadMap (input to errorSec computation):")
                print(f"    Python radius_max: {py_radius_max:.6e}")
                print(f"    MATLAB radius_max: {ml_radius_max:.6e}")
                print(f"    Difference: {z_diff:.6e} ({z_rel*100:.4f}%)")
                if z_rel > 0.1:
                    print(f"    [WARNING] Large difference in Z!")
        
        print()
