"""Compare detailed reduction values between Python and MATLAB"""

import pickle
import scipy.io
import numpy as np

# Load Python log
print("Loading Python log...")
with open('upstream_python_log.pkl', 'rb') as f:
    py_data = pickle.load(f)

py_entries = py_data.get('upstreamLog', [])
print(f"Python entries: {len(py_entries)}")

# Load MATLAB log
print("Loading MATLAB log...")
ml_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=True)
ml_entries = ml_data['upstreamLog']
print(f"MATLAB entries: {ml_entries.size}")

# Find Step 3 entries
step = 3
py_step3 = [e for e in py_entries if e.get('step') == step]
ml_step3_indices = [i for i in range(ml_entries.size) if ml_entries[i, 0]['step'][0, 0] == step]

print(f"\nStep {step} entries:")
print(f"  Python: {len(py_step3)}")
print(f"  MATLAB: {len(ml_step3_indices)}")

if py_step3 and ml_step3_indices:
    py = py_step3[-1]
    ml_idx = ml_step3_indices[-1]
    ml = ml_entries[ml_idx, 0]
    
    print(f"\n{'='*80}")
    print(f"COMPARING REDUCTION DETAILS - Step {step}")
    print(f"{'='*80}\n")
    
    # Get reduction details
    py_rred = py.get('Rred_after_reduction')
    ml_rred = ml['Rred_after_reduction']
    
    if py_rred and ml_rred.size > 0:
        py_rd = py_rred.get('reduction_details')
        ml_rd_struct = ml_rred[0, 0]['reduction_details']
        
        if py_rd and ml_rd_struct.size > 0:
            ml_rd = ml_rd_struct[0, 0]
            
            print("REDUCTION DETAILS COMPARISON:\n")
            
            # Compare key values
            fields_to_compare = [
                'diagpercent',
                'dHmax',
                'redIdx',
                'dHerror',
                'final_generators',
                'h_computed_max',
                'h_computed_min',
            ]
            
            for field in fields_to_compare:
                py_val = py_rd.get(field)
                if field in ml_rd.dtype.names:
                    ml_val = ml_rd[field]
                    if ml_val.size > 0:
                        ml_val = ml_val[0, 0] if ml_val.size == 1 else ml_val
                    else:
                        ml_val = None
                else:
                    ml_val = None
                
                if py_val is not None and ml_val is not None:
                    if isinstance(py_val, np.ndarray) and isinstance(ml_val, np.ndarray):
                        if py_val.shape == ml_val.shape:
                            match = np.allclose(py_val, ml_val, rtol=1e-10, atol=1e-12)
                            print(f"{field:20s}: Python={py_val}, MATLAB={ml_val}, Match={match}")
                            if not match:
                                diff = np.abs(py_val - ml_val)
                                rel_diff = diff / (np.abs(ml_val) + 1e-15)
                                print(f"{'':20s}  Diff: {diff}, Rel: {rel_diff*100:.2f}%")
                        else:
                            print(f"{field:20s}: Shape mismatch - Python: {py_val.shape}, MATLAB: {ml_val.shape}")
                    else:
                        match = abs(py_val - ml_val) < 1e-10 if isinstance(py_val, (int, float)) and isinstance(ml_val, (int, float)) else (py_val == ml_val)
                        print(f"{field:20s}: Python={py_val}, MATLAB={ml_val}, Match={match}")
                        if not match and isinstance(py_val, (int, float)) and isinstance(ml_val, (int, float)):
                            rel_diff = abs(py_val - ml_val) / (abs(ml_val) + 1e-15) * 100
                            print(f"{'':20s}  Diff: {abs(py_val - ml_val):.10e}, Rel: {rel_diff:.2f}%")
                else:
                    print(f"{field:20s}: Python={py_val}, MATLAB={ml_val}")
            
            # Compare arrays
            print("\nARRAY COMPARISONS:\n")
            
            array_fields = ['h_initial', 'h_computed', 'redIdx_arr', 'gredIdx']
            for field in array_fields:
                py_val = py_rd.get(field)
                if field in ml_rd.dtype.names:
                    ml_val = ml_rd[field]
                    if ml_val.size > 0:
                        ml_val = ml_val[0, 0] if ml_val.ndim > 1 else ml_val
                        ml_val = np.array(ml_val).flatten()
                    else:
                        ml_val = None
                else:
                    ml_val = None
                
                if py_val is not None and ml_val is not None:
                    py_arr = np.array(py_val).flatten()
                    ml_arr = np.array(ml_val).flatten()
                    
                    if len(py_arr) == len(ml_arr):
                        match = np.allclose(py_arr, ml_arr, rtol=1e-10, atol=1e-12)
                        print(f"{field:20s}: Length={len(py_arr)}, Match={match}")
                        if not match:
                            diff = np.abs(py_arr - ml_arr)
                            max_diff_idx = np.argmax(diff)
                            print(f"{'':20s}  Max diff at idx {max_diff_idx}: {diff[max_diff_idx]:.10e}")
                            print(f"{'':20s}  Python[{max_diff_idx}]={py_arr[max_diff_idx]:.10e}")
                            print(f"{'':20s}  MATLAB[{max_diff_idx}]={ml_arr[max_diff_idx]:.10e}")
                    else:
                        print(f"{field:20s}: Length mismatch - Python: {len(py_arr)}, MATLAB: {len(ml_arr)}")
                        print(f"{'':20s}  Python: {py_arr}")
                        print(f"{'':20s}  MATLAB: {ml_arr}")
                else:
                    print(f"{field:20s}: Python={py_val is not None}, MATLAB={ml_val is not None}")
            
            # Critical comparison: final_generators
            print(f"\n{'='*80}")
            print("CRITICAL: FINAL GENERATORS")
            print(f"{'='*80}")
            py_final = py_rd.get('final_generators')
            ml_final = ml_rd['final_generators'][0, 0] if 'final_generators' in ml_rd.dtype.names else None
            print(f"Python final_generators: {py_final}")
            print(f"MATLAB final_generators: {ml_final}")
            if py_final != ml_final:
                print(f"*** MISMATCH: {py_final} vs {ml_final} ***")
                print("This is the ROOT CAUSE of the 20% difference in errorSec!")
            else:
                print("Match!")
            
        else:
            print("No reduction_details found in one or both logs")
    else:
        print("Rred_after_reduction not found in one or both logs")
else:
    print("Step 3 entries not found in one or both logs")
