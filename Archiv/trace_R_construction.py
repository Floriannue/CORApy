"""Trace how R is constructed before reduction in both Python and MATLAB"""

import pickle
import scipy.io
import numpy as np

# Load logs
print("Loading logs...")
with open('upstream_python_log.pkl', 'rb') as f:
    py_data = pickle.load(f)
ml_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=True)

py_entries = py_data.get('upstreamLog', [])
ml_entries = ml_data['upstreamLog']

# Find Step 3
step = 3
py_step3 = [e for e in py_entries if e.get('step') == step]
ml_step3_indices = [i for i in range(ml_entries.size) if ml_entries[i, 0]['step'][0, 0] == step]

if py_step3 and ml_step3_indices:
    py = py_step3[-1]
    ml_idx = ml_step3_indices[-1]
    ml = ml_entries[ml_idx, 0]
    
    print(f"\n{'='*80}")
    print(f"TRACING R CONSTRUCTION - Step {step}")
    print(f"{'='*80}\n")
    
    # Check R before reduction
    py_r_before = py.get('R_before_reduction')
    ml_r_before = ml['R_before_reduction']
    
    print("R BEFORE REDUCTION (the input to reduce):")
    if py_r_before and ml_r_before.size > 0:
        ml_r = ml_r_before[0, 0]
        
        py_num_gen = py_r_before.get('num_generators')
        ml_num_gen = ml_r['num_generators'][0, 0] if 'num_generators' in ml_r.dtype.names else None
        
        print(f"  Python: {py_num_gen} generators")
        print(f"  MATLAB: {ml_num_gen} generators")
        print(f"  Difference: {abs(py_num_gen - ml_num_gen) if py_num_gen and ml_num_gen else 'N/A'} generators\n")
        
        # Check if Rdelta is tracked
        py_rdelta_num = py_r_before.get('Rdelta_num_generators')
        ml_rdelta_num = ml_r['Rdelta_num_generators'][0, 0] if 'Rdelta_num_generators' in ml_r.dtype.names else None
        
        if py_rdelta_num is not None or ml_rdelta_num is not None:
            print("RDELTA (input to priv_abstractionError_adaptive):")
            print(f"  Python: {py_rdelta_num} generators")
            print(f"  MATLAB: {ml_rdelta_num} generators")
            if py_rdelta_num and ml_rdelta_num:
                print(f"  Difference: {abs(py_rdelta_num - ml_rdelta_num)} generators")
                print(f"  R = reduce(Rdelta) should preserve or reduce generators")
                print(f"  Python: Rdelta={py_rdelta_num} -> R={py_num_gen}")
                print(f"  MATLAB: Rdelta={ml_rdelta_num} -> R={ml_num_gen}")
        
        # Compare centers
        py_center = py_r_before.get('center')
        ml_center = ml_r['center'] if 'center' in ml_r.dtype.names else None
        
        if py_center is not None:
            py_center_arr = np.array(py_center).flatten()
            print(f"\n  Python center shape: {py_center_arr.shape}, values: {py_center_arr}")
        
        if ml_center is not None and ml_center.size > 0:
            ml_center_arr = np.array(ml_center).flatten()
            print(f"  MATLAB center shape: {ml_center_arr.shape}, values: {ml_center_arr}")
            if py_center is not None:
                if py_center_arr.shape == ml_center_arr.shape:
                    match = np.allclose(py_center_arr, ml_center_arr, rtol=1e-10)
                    print(f"  Center match: {match}")
                else:
                    print(f"  Center shape mismatch!")
        
        # Compare generators if available
        py_gens = py_r_before.get('generators')
        ml_gens = ml_r['generators'] if 'generators' in ml_r.dtype.names else None
        
        if py_gens is not None:
            py_gens_arr = np.array(py_gens)
            print(f"\n  Python generators shape: {py_gens_arr.shape}")
            print(f"  Python generator norms: {np.linalg.norm(py_gens_arr, axis=0)}")
        
        if ml_gens is not None and ml_gens.size > 0:
            ml_gens_arr = np.array(ml_gens[0, 0]) if ml_gens.ndim > 1 else np.array(ml_gens)
            if ml_gens_arr.size > 0 and ml_gens_arr.ndim == 2:
                print(f"  MATLAB generators shape: {ml_gens_arr.shape}")
                print(f"  MATLAB generator norms: {np.linalg.norm(ml_gens_arr, axis=0)}")
            else:
                print(f"  MATLAB generators: Not properly stored (shape: {ml_gens_arr.shape})")
    
    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print("  R is the result of: R = reduce(Rdelta, 'adaptive', ...)")
    print("  If Rdelta has different generator counts, R will also differ.")
    print("  Need to trace further upstream to find where Rdelta diverges.")
