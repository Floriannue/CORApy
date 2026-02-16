"""Compare Rdelta between Python and MATLAB to find where generator count diverges"""

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
    print(f"COMPARING RDELTA - Step {step}")
    print(f"{'='*80}\n")
    
    # Check R before reduction (this is Rmax = reduce(Rdelta, ...))
    py_r_before = py.get('R_before_reduction')
    ml_r_before = ml['R_before_reduction']
    
    print("R BEFORE REDUCTION (Rmax = reduce(Rdelta, ...)):")
    if py_r_before and ml_r_before.size > 0:
        ml_r = ml_r_before[0, 0]
        
        py_num_gen = py_r_before.get('num_generators')
        ml_num_gen = ml_r['num_generators'][0, 0] if 'num_generators' in ml_r.dtype.names else None
        
        print(f"  Python: {py_num_gen} generators")
        print(f"  MATLAB: {ml_num_gen} generators")
        print(f"  Difference: {abs(py_num_gen - ml_num_gen) if py_num_gen and ml_num_gen else 'N/A'} generators\n")
        
        # Try to get generators
        py_gens = py_r_before.get('generators')
        ml_gens = ml_r['generators'] if 'generators' in ml_r.dtype.names else None
        
        if py_gens is not None:
            py_gens_arr = np.array(py_gens)
            print(f"  Python generators shape: {py_gens_arr.shape}")
            if py_gens_arr.size > 0:
                print(f"  Python generator norms: {np.linalg.norm(py_gens_arr, axis=0)}")
        
        if ml_gens is not None and ml_gens.size > 0:
            try:
                ml_gens_arr = np.array(ml_gens[0, 0]) if ml_gens.ndim > 1 else np.array(ml_gens)
                if ml_gens_arr.size > 0 and ml_gens_arr.ndim == 2:
                    print(f"  MATLAB generators shape: {ml_gens_arr.shape}")
                    print(f"  MATLAB generator norms: {np.linalg.norm(ml_gens_arr, axis=0)}")
                else:
                    print(f"  MATLAB generators: shape={ml_gens_arr.shape}, size={ml_gens_arr.size}")
            except Exception as e:
                print(f"  MATLAB generators: Error accessing - {e}")
    
    # Check if Rdelta is tracked
    print(f"\n{'='*80}")
    print("RDELTA TRACKING:")
    
    # Check in options or upstream log
    # Rdelta should be tracked in linReach_adaptive before the first reduction
    print("  Note: Rdelta tracking needs to be added to linReach_adaptive")
    print("  Rdelta = Rstart + (-nlnsys.linError.p.x)")
    print("  Then R = reduce(Rdelta, 'adaptive', ...)")
    print("  The difference in R generator count suggests Rdelta differs")
