"""Investigate why Python has 8 generators while MATLAB has 13"""

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
    print(f"INVESTIGATING GENERATOR COUNT MISMATCH - Step {step}")
    print(f"{'='*80}\n")
    
    # Check R before reduction
    py_r_before = py.get('R_before_reduction')
    ml_r_before = ml['R_before_reduction']
    
    print("R BEFORE REDUCTION:")
    if py_r_before and ml_r_before.size > 0:
        ml_r = ml_r_before[0, 0]
        
        py_num_gen = py_r_before.get('num_generators')
        ml_num_gen = ml_r['num_generators'][0, 0] if 'num_generators' in ml_r.dtype.names else None
        
        print(f"  Python: {py_num_gen} generators")
        print(f"  MATLAB: {ml_num_gen} generators")
        diff_val = abs(py_num_gen - ml_num_gen) if py_num_gen and ml_num_gen else 'N/A'
        print(f"  Difference: {diff_val} generators")
        
        # Try to get generators
        py_gens = py_r_before.get('generators')
        ml_gens = ml_r['generators'] if 'generators' in ml_r.dtype.names else None
        
        if py_gens is not None:
            py_gens_arr = np.array(py_gens)
            print(f"\n  Python generators shape: {py_gens_arr.shape}")
            print(f"  Python generators norm: {np.linalg.norm(py_gens_arr, axis=0)}")
        
        if ml_gens is not None and ml_gens.size > 0:
            ml_gens_arr = np.array(ml_gens[0, 0])
            print(f"\n  MATLAB generators shape: {ml_gens_arr.shape}")
            if ml_gens_arr.size > 0 and ml_gens_arr.ndim > 0:
                if ml_gens_arr.ndim == 2:
                    print(f"  MATLAB generators norm: {np.linalg.norm(ml_gens_arr, axis=0)}")
                else:
                    print(f"  MATLAB generators: {ml_gens_arr}")
            else:
                print(f"  MATLAB generators: Empty or scalar")
        
        # Check center
        py_center = py_r_before.get('center')
        ml_center = ml_r['center'] if 'center' in ml_r.dtype.names else None
        
        if py_center is not None:
            py_center_arr = np.array(py_center)
            print(f"\n  Python center: {py_center_arr.flatten()}")
        
        if ml_center is not None and ml_center.size > 0:
            ml_center_arr = np.array(ml_center[0, 0])
            print(f"  MATLAB center: {ml_center_arr.flatten()}")
            if py_center is not None:
                match = np.allclose(py_center_arr.flatten(), ml_center_arr.flatten(), rtol=1e-10)
                print(f"  Center match: {match}")
    
    # Check Rred after reduction
    print(f"\n{'='*80}")
    print("RRED AFTER REDUCTION:")
    py_rred = py.get('Rred_after_reduction')
    ml_rred = ml['Rred_after_reduction']
    
    if py_rred and ml_rred.size > 0:
        ml_rred_struct = ml_rred[0, 0]
        
        py_num_gen = py_rred.get('num_generators')
        ml_num_gen = ml_rred_struct['num_generators'][0, 0] if 'num_generators' in ml_rred_struct.dtype.names else None
        
        print(f"  Python: {py_num_gen} generators")
        print(f"  MATLAB: {ml_num_gen} generators")
        diff_val = abs(py_num_gen - ml_num_gen) if py_num_gen and ml_num_gen else 'N/A'
        print(f"  Difference: {diff_val} generators")
        
        # Check reduction details
        py_rd = py_rred.get('reduction_details')
        ml_rd_struct = ml_rred_struct['reduction_details'] if 'reduction_details' in ml_rred_struct.dtype.names else None
        
        if py_rd and ml_rd_struct is not None and ml_rd_struct.size > 0:
            ml_rd = ml_rd_struct[0, 0]
            
            print(f"\n  Python redIdx: {py_rd.get('redIdx')}")
            print(f"  MATLAB redIdx: {ml_rd['redIdx'][0, 0] if 'redIdx' in ml_rd.dtype.names else 'N/A'}")
            
            print(f"\n  Python final_generators: {py_rd.get('final_generators')}")
            print(f"  MATLAB final_generators: {ml_rd['final_generators'][0, 0] if 'final_generators' in ml_rd.dtype.names else 'N/A'}")
    
    # Check Z before quadMap (this is cartProd(Rred, U))
    print(f"\n{'='*80}")
    print("Z BEFORE QUADMAP (cartProd(Rred, U)):")
    py_z = py.get('Z_before_quadmap')
    ml_z = ml['Z_before_quadmap']
    
    if py_z and ml_z.size > 0:
        ml_z_struct = ml_z[0, 0]
        
        # Try to get generator count from radius or other fields
        if isinstance(py_z, dict):
            py_radius = py_z.get('radius_max')
            print(f"  Python radius_max: {py_radius}")
        
        if 'radius_max' in ml_z_struct.dtype.names:
            ml_radius = ml_z_struct['radius_max'][0, 0]
            print(f"  MATLAB radius_max: {ml_radius}")
    
    print(f"\n{'='*80}")
    print("CONCLUSION:")
    diff = abs(py_num_gen - ml_num_gen) if py_num_gen and ml_num_gen else 'N/A'
    print(f"  The {diff} generator difference")
    print(f"  before reduction suggests different upstream processing.")
    print(f"  This could be due to:")
    print(f"    1. Different cartProd results")
    print(f"    2. Different intermediate reductions")
    print(f"    3. Different generator ordering/selection")
    print(f"    4. Missing/extra generators in upstream steps")
