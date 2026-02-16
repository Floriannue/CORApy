"""Deep comparison of reduction algorithm between Python and MATLAB"""

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
    print(f"DEEP REDUCTION COMPARISON - Step {step}")
    print(f"{'='*80}\n")
    
    # Get R before reduction
    py_r_before = py.get('R_before_reduction')
    ml_r_before = ml['R_before_reduction']
    
    print("R BEFORE REDUCTION:")
    if py_r_before and ml_r_before.size > 0:
        ml_r = ml_r_before[0, 0]
        py_num_gen = py_r_before.get('num_generators')
        ml_num_gen = ml_r['num_generators'][0, 0] if 'num_generators' in ml_r.dtype.names else None
        py_redFactor = py_r_before.get('redFactor')
        ml_redFactor = ml_r['redFactor'][0, 0] if 'redFactor' in ml_r.dtype.names else None
        py_diagpercent = py_r_before.get('diagpercent')
        ml_diagpercent = ml_r['diagpercent'][0, 0] if 'diagpercent' in ml_r.dtype.names else None
        
        print(f"  Num generators: Python={py_num_gen}, MATLAB={ml_num_gen}, Match={py_num_gen == ml_num_gen}")
        print(f"  redFactor: Python={py_redFactor}, MATLAB={ml_redFactor}, Match={py_redFactor == ml_redFactor if py_redFactor and ml_redFactor else False}")
        print(f"  diagpercent: Python={py_diagpercent}, MATLAB={ml_diagpercent}, Match={abs(py_diagpercent - ml_diagpercent) < 1e-15 if py_diagpercent and ml_diagpercent else False}")
        
        # Compare generators
        py_gens = py_r_before.get('generators')
        ml_gens = ml_r['generators'] if 'generators' in ml_r.dtype.names else None
        if py_gens is not None and ml_gens is not None:
            py_gens_arr = np.array(py_gens)
            ml_gens_arr = np.array(ml_gens[0, 0]) if ml_gens.size > 0 else None
            if ml_gens_arr is not None:
                if py_gens_arr.shape == ml_gens_arr.shape:
                    match = np.allclose(py_gens_arr, ml_gens_arr, rtol=1e-10, atol=1e-12)
                    print(f"  Generators shape: Python={py_gens_arr.shape}, MATLAB={ml_gens_arr.shape}, Match={match}")
                    if not match:
                        diff = np.abs(py_gens_arr - ml_gens_arr)
                        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                        print(f"    Max diff at {max_diff_idx}: {diff[max_diff_idx]:.10e}")
                else:
                    print(f"  Generators shape mismatch: Python={py_gens_arr.shape}, MATLAB={ml_gens_arr.shape}")
    
    print()
    
    # Get reduction details
    py_rred = py.get('Rred_after_reduction')
    ml_rred = ml['Rred_after_reduction']
    
    if py_rred and ml_rred.size > 0:
        py_rd = py_rred.get('reduction_details')
        ml_rd_struct = ml_rred[0, 0]['reduction_details']
        
        if py_rd and ml_rd_struct.size > 0:
            ml_rd = ml_rd_struct[0, 0]
            
            print("REDUCTION PARAMETERS:\n")
            
            # Compare dHmax
            py_dHmax = py_rd.get('dHmax')
            ml_dHmax = ml_rd['dHmax'][0, 0] if 'dHmax' in ml_rd.dtype.names and ml_rd['dHmax'].size > 0 else None
            print(f"dHmax: Python={py_dHmax}, MATLAB={ml_dHmax}")
            if py_dHmax and ml_dHmax:
                match = abs(py_dHmax - ml_dHmax) < 1e-10
                print(f"  Match: {match}")
                if not match:
                    rel_diff = abs(py_dHmax - ml_dHmax) / abs(ml_dHmax) * 100
                    print(f"  Relative difference: {rel_diff:.2f}%")
            
            print()
            print("REDUCTION DECISION:\n")
            
            # Compare h_computed in detail
            py_h_computed = py_rd.get('h_computed')
            ml_h_computed = ml_rd['h_computed'] if 'h_computed' in ml_rd.dtype.names else None
            
            if py_h_computed is not None:
                py_h_arr = np.array(py_h_computed).flatten()
                print(f"Python h_computed: shape={py_h_arr.shape}, values={py_h_arr}")
                print(f"  All <= dHmax? {np.all(py_h_arr <= py_dHmax) if py_dHmax else 'N/A'}")
                if py_dHmax:
                    py_redIdx_arr = np.where(py_h_arr <= py_dHmax)[0]
                    print(f"  Indices where h <= dHmax: {py_redIdx_arr}")
                    print(f"  Last index: {py_redIdx_arr[-1] if len(py_redIdx_arr) > 0 else 'None'}")
                    print(f"  redIdx (from tracking): {py_rd.get('redIdx')}")
            
            if ml_h_computed is not None and ml_h_computed.size > 0:
                ml_h_arr = np.array(ml_h_computed[0, 0]).flatten() if ml_h_computed.ndim > 1 else np.array(ml_h_computed).flatten()
                print(f"\nMATLAB h_computed: shape={ml_h_arr.shape}, values={ml_h_arr}")
                print(f"  All <= dHmax? {np.all(ml_h_arr <= ml_dHmax) if ml_dHmax else 'N/A'}")
                if ml_dHmax:
                    ml_redIdx_arr = np.where(ml_h_arr <= ml_dHmax)[0]
                    print(f"  Indices where h <= dHmax: {ml_redIdx_arr}")
                    print(f"  Last index: {ml_redIdx_arr[-1] if len(ml_redIdx_arr) > 0 else 'None'}")
                    print(f"  redIdx (from tracking): {ml_rd['redIdx'][0, 0] if 'redIdx' in ml_rd.dtype.names else 'N/A'}")
            
            print()
            print("KEY INSIGHT:")
            print(f"  Python finds {py_rd.get('redIdx', 'N/A')} generators can be reduced")
            print(f"  MATLAB finds {ml_rd['redIdx'][0, 0] if 'redIdx' in ml_rd.dtype.names else 'N/A'} generators can be reduced")
            print(f"  This difference causes Python to have {py_rd.get('final_generators', 'N/A')} final generators")
            print(f"  while MATLAB has {ml_rd['final_generators'][0, 0] if 'final_generators' in ml_rd.dtype.names else 'N/A'} final generators")
