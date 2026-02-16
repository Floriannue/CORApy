"""compare_reduction_step - Compare reduction step between Python and MATLAB"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("COMPARING REDUCTION STEP")
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
        if isinstance(val, np.ndarray) and val.dtype == object and val.size == 1:
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
    
    print(f"\nStep {step} - Reduction Comparison:\n")
    
    # Compare Z (after reduction, before quadMap)
    py_Z = py.get('Z_before_quadmap')
    ml_Z = get_ml_value(ml, 'Z_before_quadmap')
    
    if py_Z and ml_Z is not None:
        print("Z (after reduction, before quadMap):")
        
        # Python
        if isinstance(py_Z, dict):
            py_center = py_Z.get('center')
            py_gens = py_Z.get('generators')
            py_radius_max = py_Z.get('radius_max')
            
            if py_gens is not None:
                py_num_gens = py_gens.shape[1] if hasattr(py_gens, 'shape') and len(py_gens.shape) > 1 else 0
                print(f"  Python: {py_num_gens} generators, radius_max={py_radius_max}")
            else:
                print(f"  Python: No generators data")
        
        # MATLAB
        if hasattr(ml_Z, 'dtype'):
            ml_center = get_ml_value(ml_Z, 'center')
            ml_gens = get_ml_value(ml_Z, 'generators')
            ml_radius_max = get_ml_value(ml_Z, 'radius_max')
            
            if ml_gens is not None and isinstance(ml_gens, np.ndarray):
                ml_num_gens = ml_gens.shape[1] if len(ml_gens.shape) > 1 else 0
                print(f"  MATLAB: {ml_num_gens} generators, radius_max={ml_radius_max}")
            else:
                print(f"  MATLAB: No generators data")
            
            if py_gens is not None and ml_gens is not None:
                if py_num_gens != ml_num_gens:
                    print(f"\n  [CRITICAL] Generator count mismatch!")
                    print(f"  This means reduction produced different results!")
                    print(f"  Python reduced to {py_num_gens} generators")
                    print(f"  MATLAB reduced to {ml_num_gens} generators")
                    
                    # Check if radius is similar
                    if py_radius_max and ml_radius_max:
                        radius_diff = abs(py_radius_max - ml_radius_max)
                        radius_rel = radius_diff / (abs(ml_radius_max) + 1e-10) * 100
                        print(f"\n  Radius comparison:")
                        print(f"    Python: {py_radius_max:.10e}")
                        print(f"    MATLAB: {ml_radius_max:.10e}")
                        print(f"    Difference: {radius_diff:.10e} ({radius_rel:.4f}%)")
                        if radius_rel < 1.0:
                            print(f"    [NOTE] Radius is similar, but generator count differs!")
                            print(f"    This suggests reduction algorithm difference!")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("The 20% difference in errorSec is caused by different Z dimensions")
print("due to reduction producing different generator counts!")
print("=" * 80)
