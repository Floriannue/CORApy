"""compare_upstream_detailed - Compare Z, errorSec, and errorLagr to find divergence source"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("DETAILED UPSTREAM COMPARISON: Z, errorSec, errorLagr")
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

# Compare Z, errorSec, errorLagr for Steps 1-5
if python_upstream and len(matlab_upstream) > 0:
    print("\n" + "=" * 80)
    print("COMPARING Z, errorSec, errorLagr")
    print("=" * 80)
    
    # Group Python entries by (step, run) and take the LAST entry for each
    python_by_step_run = {}
    for e in python_upstream:
        step = e.get('step', 0)
        run = e.get('run', 0)
        key = (step, run)
        if key not in python_by_step_run:
            python_by_step_run[key] = []
        python_by_step_run[key].append(e)
    
    python_final_entries = {}
    for key, entries in python_by_step_run.items():
        python_final_entries[key] = entries[-1]  # Last entry is converged
    
    # Group MATLAB entries by (step, run)
    matlab_by_step_run = {}
    for i in range(len(matlab_upstream)):
        e = matlab_upstream[i]
        if hasattr(e, 'dtype') and 'step' in e.dtype.names and 'run' in e.dtype.names:
            step_val = e['step']
            step = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val
            run_val = e['run']
            run = run_val.item() if isinstance(run_val, np.ndarray) and run_val.size == 1 else run_val
            key = (int(step), int(run))
            matlab_by_step_run[key] = i
    
    # Find entries with Z, errorSec, errorLagr tracking
    # These are tracked when tensorOrder == 3, and run might be 0 or None
    python_with_z = {}
    for key, entry in python_final_entries.items():
        if entry.get('Z_before_quadmap') is not None:
            step = entry.get('step', 0)
            # Use step as key, ignore run for Z tracking entries
            python_with_z[step] = entry
    
    # Find MATLAB entries with Z tracking (run might be 0 or 1)
    matlab_with_z = {}
    for i in range(len(matlab_upstream)):
        e = matlab_upstream[i]
        if hasattr(e, 'dtype') and 'Z_before_quadmap' in e.dtype.names:
            step_val = e['step']
            step = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val
            matlab_with_z[int(step)] = i
    
    # Find common steps
    common_steps = sorted(set(python_with_z.keys()) & set(matlab_with_z.keys()))
    
    print(f"\nFound {len(common_steps)} common steps with Z tracking")
    print(f"Comparing Steps 1-5 to find where divergence starts:\n")
    
    for step in common_steps[:5]:
        py = python_with_z[step]
        ml_idx = matlab_with_z[step]
        ml = matlab_upstream[ml_idx]
        
        print(f"{'='*80}")
        print(f"Step {step} (with Z, errorSec, errorLagr tracking):")
        print(f"{'='*80}")
        
        # Compare Z before quadMap
        py_z = py.get('Z_before_quadmap') if isinstance(py, dict) else None
        ml_z_val = get_ml_value(ml, 'Z_before_quadmap')
        if py_z and ml_z_val is not None:
            py_radius_max = py_z.get('radius_max') if isinstance(py_z, dict) else None
            ml_radius_max = get_ml_value(ml_z_val, 'radius_max') if ml_z_val is not None else None
            if py_radius_max is not None and ml_radius_max is not None:
                z_diff = abs(py_radius_max - ml_radius_max)
                z_rel = z_diff / max(abs(ml_radius_max), 1e-10)
                print(f"Z before quadMap:")
                print(f"  Python radius_max: {py_radius_max:.6e}")
                print(f"  MATLAB radius_max: {ml_radius_max:.6e}")
                print(f"  Difference: {z_diff:.6e} ({z_rel*100:.4f}%)")
        
        # Compare errorSec after quadMap
        py_es = py.get('errorSec_before_combine') if isinstance(py, dict) else None
        ml_es_val = get_ml_value(ml, 'errorSec_before_combine')
        if py_es and ml_es_val is not None:
            py_radius_max = py_es.get('radius_max') if isinstance(py_es, dict) else None
            ml_radius_max = get_ml_value(ml_es_val, 'radius_max') if ml_es_val is not None else None
            if py_radius_max is not None and ml_radius_max is not None:
                es_diff = abs(py_radius_max - ml_radius_max)
                es_rel = es_diff / max(abs(ml_radius_max), 1e-10)
                print(f"\nerrorSec after quadMap (before combine):")
                print(f"  Python radius_max: {py_radius_max:.6e}")
                print(f"  MATLAB radius_max: {ml_radius_max:.6e}")
                print(f"  Difference: {es_diff:.6e} ({es_rel*100:.4f}%)")
                if es_rel > 0.1:
                    print(f"  [WARNING] Large difference in errorSec!")
        
        # Compare errorLagr before combine
        py_el = py.get('errorLagr_before_combine') if isinstance(py, dict) else None
        ml_el_val = get_ml_value(ml, 'errorLagr_before_combine')
        if py_el and ml_el_val is not None:
            py_radius_max = py_el.get('radius_max') if isinstance(py_el, dict) else None
            ml_radius_max = get_ml_value(ml_el_val, 'radius_max') if ml_el_val is not None else None
            if py_radius_max is not None and ml_radius_max is not None:
                el_diff = abs(py_radius_max - ml_radius_max)
                el_rel = el_diff / max(abs(ml_radius_max), 1e-10)
                print(f"\nerrorLagr before combine:")
                print(f"  Python radius_max: {py_radius_max:.6e}")
                print(f"  MATLAB radius_max: {ml_radius_max:.6e}")
                print(f"  Difference: {el_diff:.6e} ({el_rel*100:.4f}%)")
                if el_rel > 0.1:
                    print(f"  [WARNING] Large difference in errorLagr!")
        
        # Compare VerrorDyn before reduce (errorSec + errorLagr)
        py_vd = py.get('VerrorDyn_before_reduce') if isinstance(py, dict) else None
        ml_vd_val = get_ml_value(ml, 'VerrorDyn_before_reduce')
        if py_vd and ml_vd_val is not None:
            py_radius_max = py_vd.get('radius_max') if isinstance(py_vd, dict) else None
            ml_radius_max = get_ml_value(ml_vd_val, 'radius_max') if ml_vd_val is not None else None
            if py_radius_max is not None and ml_radius_max is not None:
                vd_diff = abs(py_radius_max - ml_radius_max)
                vd_rel = vd_diff / max(abs(ml_radius_max), 1e-10)
                print(f"\nVerrorDyn before reduce (errorSec + errorLagr):")
                print(f"  Python radius_max: {py_radius_max:.6e}")
                print(f"  MATLAB radius_max: {ml_radius_max:.6e}")
                print(f"  Difference: {vd_diff:.6e} ({vd_rel*100:.4f}%)")
                if vd_rel > 0.1:
                    print(f"  [WARNING] Large difference in VerrorDyn!")
        
        # Compare VerrorDyn after reduce
        py_vd_after = py.get('VerrorDyn_after_reduce') if isinstance(py, dict) else None
        ml_vd_after_val = get_ml_value(ml, 'VerrorDyn_after_reduce')
        if py_vd_after and ml_vd_after_val is not None:
            py_radius_max = py_vd_after.get('radius_max') if isinstance(py_vd_after, dict) else None
            ml_radius_max = get_ml_value(ml_vd_after_val, 'radius_max') if ml_vd_after_val is not None else None
            if py_radius_max is not None and ml_radius_max is not None:
                vd_diff = abs(py_radius_max - ml_radius_max)
                vd_rel = vd_diff / max(abs(ml_radius_max), 1e-10)
                print(f"\nVerrorDyn after reduce:")
                print(f"  Python radius_max: {py_radius_max:.6e}")
                print(f"  MATLAB radius_max: {ml_radius_max:.6e}")
                print(f"  Difference: {vd_diff:.6e} ({vd_rel*100:.4f}%)")
                if vd_rel > 0.1:
                    print(f"  [WARNING] Large difference in VerrorDyn after reduce!")
        
        print()

print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print("Look for where the divergence first appears:")
print("1. If Z differs -> issue in reachable set computation")
print("2. If errorSec differs -> issue in quadMap computation")
print("3. If errorLagr differs -> issue in third-order tensor computation")
print("4. If VerrorDyn differs but components don't -> issue in combination")
print("=" * 80)
