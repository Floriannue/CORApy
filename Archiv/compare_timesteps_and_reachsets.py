"""compare_timesteps_and_reachsets - Compare time steps and reachable set sizes"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("COMPARING TIME STEPS AND REACHABLE SET SIZES")
print("=" * 80)

# Load Python optimaldeltat log
python_file = 'upstream_python_log.pkl'
if os.path.exists(python_file):
    with open(python_file, 'rb') as f:
        python_data = pickle.load(f)
    python_optimaldeltat = python_data.get('optimaldeltatLog', [])
    print(f"\n[OK] Loaded Python optimaldeltat log: {len(python_optimaldeltat)} entries")
else:
    print(f"\n[ERROR] Python log file not found: {python_file}")
    python_optimaldeltat = []

# Load MATLAB optimaldeltat log
matlab_file = 'optimaldeltat_matlab_log.mat'
if os.path.exists(matlab_file):
    try:
        matlab_data = scipy.io.loadmat(matlab_file, squeeze_me=True)
        if 'optimaldeltatLog' in matlab_data:
            matlab_log = matlab_data['optimaldeltatLog']
            # Convert structured array to list
            if isinstance(matlab_log, np.ndarray) and len(matlab_log.shape) > 0:
                matlab_optimaldeltat = []
                for i in range(len(matlab_log)):
                    entry = {}
                    for field in matlab_log.dtype.names:
                        val = matlab_log[field][i]
                        if isinstance(val, np.ndarray):
                            if val.size == 1:
                                try:
                                    entry[field] = float(val.item())
                                except (ValueError, TypeError):
                                    entry[field] = val.item()
                            else:
                                entry[field] = val.flatten()
                        else:
                            entry[field] = val
                    matlab_optimaldeltat.append(entry)
            else:
                matlab_optimaldeltat = []
        else:
            matlab_optimaldeltat = []
        print(f"[OK] Loaded MATLAB optimaldeltat log: {len(matlab_optimaldeltat)} entries")
    except Exception as e:
        print(f"[ERROR] Could not load MATLAB log: {e}")
        matlab_optimaldeltat = []
else:
    print(f"[WARNING] MATLAB optimaldeltat log file not found: {matlab_file}")
    matlab_optimaldeltat = []

# Compare time step selections
if python_optimaldeltat and matlab_optimaldeltat:
    print("\n" + "=" * 80)
    print("TIME STEP SELECTION COMPARISON")
    print("=" * 80)
    
    # Group by step
    python_by_step = {}
    for e in python_optimaldeltat:
        step = e.get('step', 0)
        if step not in python_by_step:
            python_by_step[step] = []
        python_by_step[step].append(e)
    
    matlab_by_step = {}
    for e in matlab_optimaldeltat:
        step = e.get('step', 0)
        if step not in matlab_by_step:
            matlab_by_step[step] = []
        matlab_by_step[step].append(e)
    
    # Compare first 20 steps
    common_steps = sorted(set(python_by_step.keys()) & set(matlab_by_step.keys()))
    num_compare = min(20, len(common_steps))
    
    print(f"\nComparing first {num_compare} steps:\n")
    print(f"{'Step':<6} {'Python dt':<15} {'MATLAB dt':<15} {'Diff %':<10} {'Python rR':<15} {'MATLAB rR':<15} {'rR Diff %':<10}")
    print("-" * 100)
    
    for step in common_steps[:num_compare]:
        # Take last entry for each step (converged)
        py_entries = python_by_step[step]
        ml_entries = matlab_by_step[step]
        
        if py_entries and ml_entries:
            py = py_entries[-1]
            ml = ml_entries[-1]
            
            py_dt = py.get('deltatest', 0)
            ml_dt = ml.get('deltatest', 0)
            dt_diff = abs(py_dt - ml_dt) / max(abs(ml_dt), 1e-10) * 100 if ml_dt != 0 else 0
            
            py_rR = py.get('rR', 0)
            ml_rR = ml.get('rR', 0)
            rR_diff = abs(py_rR - ml_rR) / max(abs(ml_rR), 1e-10) * 100 if ml_rR != 0 else 0
            
            print(f"{step:<6} {py_dt:<15.6e} {ml_dt:<15.6e} {dt_diff:<10.2f} {py_rR:<15.6e} {ml_rR:<15.6e} {rR_diff:<10.2f}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("If Python's time steps are larger -> larger reachable sets -> larger Z -> larger errorSec")
print("But we see Python's VerrorDyn is SMALLER, which suggests:")
print("1. Python's time steps might actually be smaller (check above)")
print("2. Or Python's reduction is more aggressive")
print("3. Or Python's errorLagr (third-order term) is much smaller")
