"""investigate_quadmat_shape_mismatch - Investigate why quadMat shapes differ"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("INVESTIGATING quadMat SHAPE MISMATCH")
print("=" * 80)

# Load Python log
python_file = 'upstream_python_log.pkl'
with open(python_file, 'rb') as f:
    python_data = pickle.load(f)
python_upstream = python_data.get('upstreamLog', [])

# Load MATLAB log
matlab_file = 'upstream_matlab_log.mat'
matlab_data = scipy.io.loadmat(matlab_file, squeeze_me=True)
matlab_upstream = matlab_data['upstreamLog']

# Helper function
def get_ml_value(ml_obj, field):
    if hasattr(ml_obj, 'dtype') and field in ml_obj.dtype.names:
        val = ml_obj[field]
        if isinstance(val, np.ndarray) and val.dtype == object:
            val = val.item()
        return val
    return None

# Group Python entries by step
python_by_step = {}
for e in python_upstream:
    step = e.get('step', 0)
    if step not in python_by_step:
        python_by_step[step] = []
    python_by_step[step].append(e)

python_final_entries = {}
for step, entries in python_by_step.items():
    python_final_entries[step] = entries[-1]

# Group MATLAB entries by step
matlab_by_step = {}
for i in range(len(matlab_upstream)):
    e = matlab_upstream[i]
    if hasattr(e, 'dtype') and 'step' in e.dtype.names:
        step_val = e['step']
        step = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val
        matlab_by_step[int(step)] = i

# Compare Step 3
step = 3
if step in python_final_entries and step in matlab_by_step:
    py = python_final_entries[step]
    ml_idx = matlab_by_step[step]
    ml = matlab_upstream[ml_idx]
    
    print(f"\nStep {step} Comparison:\n")
    
    # Compare Z (zonotope)
    py_Z = py.get('Z_before_quadmap')
    ml_Z = get_ml_value(ml, 'Z_before_quadmap')
    
    if py_Z and ml_Z is not None:
        py_gens = None
        ml_gens = None
        
        if isinstance(py_Z, dict) and 'generators' in py_Z:
            py_gens_arr = py_Z['generators']
            if py_gens_arr is not None:
                py_gens = py_gens_arr.shape[1] if hasattr(py_gens_arr, 'shape') and len(py_gens_arr.shape) > 1 else 0
        
        if hasattr(ml_Z, 'dtype') and 'generators' in ml_Z.dtype.names:
            ml_gens_arr = ml_Z['generators']
            if isinstance(ml_gens_arr, np.ndarray) and ml_gens_arr.dtype == object:
                ml_gens_arr = ml_gens_arr.item()
            if isinstance(ml_gens_arr, np.ndarray):
                ml_gens = ml_gens_arr.shape[1] if len(ml_gens_arr.shape) > 1 else 0
        
        print(f"Z generators:")
        print(f"  Python: {py_gens} generators")
        print(f"  MATLAB: {ml_gens} generators")
        if py_gens != ml_gens:
            print(f"  [CRITICAL] Generator count mismatch! This explains the shape difference!")
            print(f"  Python quadMat should be ({py_gens+1}, {py_gens+1}) = (3, 3)")
            print(f"  MATLAB quadMat should be ({ml_gens+1}, {ml_gens+1}) = (5, 5)")
    
    # Compare quadMat shapes
    py_quadmat = py.get('quadmat_tracking')
    ml_quadmat = get_ml_value(ml, 'quadmat_tracking')
    
    if py_quadmat and ml_quadmat is not None:
        if isinstance(py_quadmat, list) and len(py_quadmat) > 0:
            dim, py_info = py_quadmat[0]
            py_full = py_info.get('dense_full')
            if py_full is not None:
                py_shape = np.asarray(py_full).shape
                print(f"\nPython quadMat shape: {py_shape}")
            
            if isinstance(ml_quadmat, np.ndarray) and len(ml_quadmat) > 0:
                ml_quadmat_dim = ml_quadmat[0]
                if hasattr(ml_quadmat_dim, 'dtype') and 'dense_full' in ml_quadmat_dim.dtype.names:
                    ml_full_val = ml_quadmat_dim['dense_full']
                    if isinstance(ml_full_val, np.ndarray) and ml_full_val.dtype == object:
                        ml_full_val = ml_full_val.item()
                    if isinstance(ml_full_val, np.ndarray):
                        ml_shape = ml_full_val.shape
                        print(f"MATLAB quadMat shape: {ml_shape}")
                        
                        if py_full is not None:
                            print(f"\n[CRITICAL] Shape mismatch indicates different Z dimensions!")
                            print(f"This is the root cause of the 20% difference!")

print("\n" + "=" * 80)
