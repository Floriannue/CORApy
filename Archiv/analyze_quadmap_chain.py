"""analyze_quadmap_chain - Analyze the complete chain from H to errorSec"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("ANALYZING COMPLETE CHAIN: H -> quadMat -> errorSec")
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

# Find common steps
common_steps = sorted(set(python_final_entries.keys()) & set(matlab_by_step.keys()))

print(f"\nFound {len(common_steps)} common steps")
print(f"Analyzing Steps 1-5:\n")

for step in common_steps[:5]:
    py = python_final_entries[step]
    ml_idx = matlab_by_step[step]
    ml = matlab_upstream[ml_idx]
    
    print(f"{'='*80}")
    print(f"Step {step}:")
    print(f"{'='*80}")
    
    # 1. Compare H (Hessian)
    print("\n1. HESSIAN (H):")
    py_H = py.get('H_before_quadmap') if isinstance(py, dict) else None
    ml_H = get_ml_value(ml, 'H_before_quadmap')
    
    if py_H and ml_H is not None:
        print(f"   Python H length: {len(py_H) if py_H else 0}")
        if py_H and len(py_H) > 0:
            h0_py = py_H[0]
            if isinstance(h0_py, dict) and 'matrix' in h0_py:
                h0_py_mat = h0_py['matrix']
                print(f"   Python H[0] type: numeric matrix")
                print(f"   Python H[0] shape: {h0_py_mat.shape if hasattr(h0_py_mat, 'shape') else 'N/A'}")
                print(f"   Python H[0] max: {np.max(np.abs(h0_py_mat)) if hasattr(h0_py_mat, '__abs__') else 'N/A'}")
    
    # 2. Compare Z (zonotope)
    print("\n2. ZONOTOPE (Z):")
    py_Z = py.get('Z_before_quadmap') if isinstance(py, dict) else None
    ml_Z = get_ml_value(ml, 'Z_before_quadmap')
    
    if py_Z and ml_Z is not None:
        py_radius_max = py_Z.get('radius_max') if isinstance(py_Z, dict) else None
        ml_radius_max = get_ml_value(ml_Z, 'radius_max')
        print(f"   Python Z radius_max: {py_radius_max}")
        print(f"   MATLAB Z radius_max: {ml_radius_max}")
        if py_radius_max and ml_radius_max:
            z_diff = abs(py_radius_max - ml_radius_max)
            z_rel = z_diff / (abs(ml_radius_max) + 1e-10) * 100
            print(f"   Z difference: {z_diff:.6e} ({z_rel:.4f}%)")
    
    # 3. Compare quadMat
    print("\n3. quadMat (Zmat' * H * Zmat):")
    py_quadmat = py.get('quadmat_tracking') if isinstance(py, dict) else None
    ml_quadmat = get_ml_value(ml, 'quadmat_tracking')
    
    if py_quadmat and isinstance(py_quadmat, list) and len(py_quadmat) > 0:
        dim, info = py_quadmat[0]
        print(f"   Python quadMat type: {info.get('type', 'unknown')}")
        print(f"   Python is_interval: {info.get('is_interval', False)}")
        print(f"   Python is_sparse: {info.get('is_sparse', False)}")
        
        if info.get('dense_diag') is not None:
            py_diag = np.asarray(info['dense_diag'])
            print(f"   Python dense_diag: {py_diag}")
            print(f"   Python dense_max: {info.get('dense_max', 'N/A')}")
            
            # MATLAB comparison
            if ml_quadmat is not None:
                if isinstance(ml_quadmat, np.ndarray) and len(ml_quadmat) > 0:
                    ml_quadmat_dim = ml_quadmat[0]
                    if hasattr(ml_quadmat_dim, 'dtype') and 'dense_diag' in ml_quadmat_dim.dtype.names:
                        ml_diag = ml_quadmat_dim['dense_diag']
                        if isinstance(ml_diag, np.ndarray):
                            ml_diag = ml_diag.flatten()
                        
                        print(f"   MATLAB dense_diag: {ml_diag}")
                        
                        if len(py_diag) == len(ml_diag):
                            diag_diff = np.abs(py_diag - ml_diag)
                            diag_rel = diag_diff / (np.abs(ml_diag) + 1e-10) * 100
                            print(f"   Diagonal difference: {diag_diff}")
                            print(f"   Relative difference: {diag_rel}%")
                            max_rel = np.max(diag_rel)
                            print(f"   Max relative difference: {max_rel:.6f}%")
                            if max_rel > 0.1:
                                print(f"   [WARNING] Significant difference in quadMat diagonal!")
    else:
        print(f"   No quadMat tracking data available")
    
    # 4. Compare errorSec
    print("\n4. errorSec (after quadMap):")
    py_errorSec = py.get('errorSec_before_combine') if isinstance(py, dict) else None
    ml_errorSec = get_ml_value(ml, 'errorSec_before_combine')
    
    if py_errorSec and ml_errorSec is not None:
        py_radius_max = py_errorSec.get('radius_max') if isinstance(py_errorSec, dict) else None
        ml_radius_max = get_ml_value(ml_errorSec, 'radius_max')
        print(f"   Python errorSec radius_max: {py_radius_max}")
        print(f"   MATLAB errorSec radius_max: {ml_radius_max}")
        if py_radius_max and ml_radius_max:
            es_diff = abs(py_radius_max - ml_radius_max)
            es_rel = es_diff / (abs(ml_radius_max) + 1e-10) * 100
            print(f"   errorSec difference: {es_diff:.6e} ({es_rel:.4f}%)")
            if es_rel > 1.0:
                print(f"   [WARNING] Large difference in errorSec!")
    
    print()

print("=" * 80)
print("CHAIN ANALYSIS COMPLETE")
print("=" * 80)
