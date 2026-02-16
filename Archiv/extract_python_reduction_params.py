"""Extract reduction parameters from Python Step 2 Run 2"""
import pickle
import numpy as np

print("=" * 80)
print("EXTRACTING PYTHON REDUCTION PARAMETERS FOR STEP 2 RUN 2")
print("=" * 80)

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 2 Run 2
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        py_entry = entry
        break

if py_entry is None:
    print("[ERROR] Could not find Python Step 2 Run 2")
    exit(1)

print("[OK] Found Python Step 2 Run 2")

# Extract reduction parameters from initReach_tracking
if 'initReach_tracking' in py_entry:
    it = py_entry['initReach_tracking']
    
    print("\n" + "=" * 80)
    print("REDUCTION PARAMETERS FROM initReach_tracking")
    print("=" * 80)
    
    # Generator counts
    rhom_tp_num = it.get('Rhom_tp_num_generators')
    rend_tp_num = it.get('Rend_tp_num_generators')
    print(f"\nGenerator counts:")
    print(f"  Rhom_tp (before reduction): {rhom_tp_num}")
    print(f"  Rend_tp (after reduction): {rend_tp_num}")
    if rhom_tp_num and rend_tp_num:
        print(f"  Reduced: {rhom_tp_num - rend_tp_num} generators")
    
    # Reduction parameters
    reduction_params = {
        'diagpercent': it.get('reduction_diagpercent'),
        'dHmax': it.get('reduction_dHmax'),
        'nrG': it.get('reduction_nrG'),
        'last0Idx': it.get('reduction_last0Idx'),
        'redIdx': it.get('reduction_redIdx'),
        'redIdx_0based': it.get('reduction_redIdx_0based'),
        'h_computed': it.get('reduction_h_computed'),
        'gredIdx': it.get('reduction_gredIdx'),
        'gredIdx_len': it.get('reduction_gredIdx_len'),
        'dHerror': it.get('reduction_dHerror')
    }
    
    print(f"\nReduction parameters:")
    for key, val in reduction_params.items():
        if val is not None:
            if isinstance(val, (list, np.ndarray)):
                arr = np.asarray(val)
                if arr.size <= 10:
                    print(f"  {key}: {val}")
                else:
                    print(f"  {key}: shape={arr.shape}, min={np.min(arr):.6e}, max={np.max(arr):.6e}")
            else:
                print(f"  {key}: {val}")
        else:
            print(f"  {key}: None")
    
    # Analyze the reduction
    if reduction_params['h_computed'] is not None and reduction_params['dHmax'] is not None:
        h = np.asarray(reduction_params['h_computed'])
        dHmax = reduction_params['dHmax']
        redIdx = reduction_params['redIdx']
        last0Idx = reduction_params['last0Idx']
        nrG = reduction_params['nrG']
        
        print(f"\n" + "=" * 80)
        print("REDUCTION ANALYSIS")
        print("=" * 80)
        print(f"Input generators: {nrG}")
        print(f"h array: {h}")
        print(f"dHmax: {dHmax}")
        print(f"h <= dHmax: {h <= dHmax}")
        print(f"Count where h <= dHmax: {np.sum(h <= dHmax)}")
        print(f"last0Idx: {last0Idx} (generators with h=0)")
        print(f"redIdx: {redIdx} (1-based index into h array)")
        print(f"redIdx_0based: {reduction_params['redIdx_0based']} (0-based index into h array)")
        
        # Compute expected final generators
        # Final = diagonal (n) + unreduced generators
        # Unreduced = nrG - last0Idx - redIdx
        if redIdx and last0Idx is not None and nrG:
            expected_unreduced = nrG - last0Idx - redIdx
            expected_final = expected_unreduced + h.shape[0]  # Add diagonal dimension
            print(f"\nExpected final generators:")
            print(f"  Unreduced from gensred: {expected_unreduced}")
            print(f"  Diagonal dimension: {h.shape[0]}")
            print(f"  Total expected: {expected_final}")
            print(f"  Actual Rend_tp: {rend_tp_num}")
            if expected_final != rend_tp_num:
                print(f"  [MISMATCH] Expected {expected_final}, got {rend_tp_num}")
            else:
                print(f"  [MATCH]")

print("\n" + "=" * 80)
print("NOTE")
print("=" * 80)
print("These parameters show how Python reduces 5->2 generators.")
print("Compare with MATLAB to find where the divergence occurs.")
