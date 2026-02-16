"""Compare Step 2 Run 2 reduction in detail - extract from reduction debug file"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("DETAILED REDUCTION COMPARISON FOR STEP 2 RUN 2")
print("=" * 80)

# Python - get reduction parameters from initReach_tracking
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        py_entry = entry
        break

if py_entry is None:
    print("[ERROR] Could not find Python Step 2 Run 2")
    exit(1)

print("[OK] Found Python Step 2 Run 2")

# Get initReach_tracking
if 'initReach_tracking' in py_entry:
    it = py_entry['initReach_tracking']
    print(f"\nPython initReach_tracking:")
    print(f"  Rhom_tp_num_generators: {it.get('Rhom_tp_num_generators')}")
    print(f"  Rend_tp_num_generators: {it.get('Rend_tp_num_generators')}")
    print(f"  redFactor: {it.get('redFactor')}")
    
    # Check reduction parameters
    reduction_keys = [k for k in it.keys() if k.startswith('reduction_')]
    if reduction_keys:
        print(f"\n  Reduction parameters found: {reduction_keys}")
        for key in reduction_keys:
            val = it.get(key)
            if val is not None:
                if isinstance(val, (list, np.ndarray)):
                    arr = np.asarray(val)
                    if arr.size <= 10:
                        print(f"    {key}: {val}")
                    else:
                        print(f"    {key}: shape={arr.shape}")
                else:
                    print(f"    {key}: {val}")
    else:
        print(f"\n  [WARNING] No reduction parameters found in initReach_tracking")
        print(f"  This means reduction details are not being captured")

# Try to get from reduction debug file
print("\n" + "=" * 80)
print("CHECKING REDUCTION DEBUG FILE")
print("=" * 80)

try:
    debug_data = scipy.io.loadmat('reduceAdaptive_debug.mat', struct_as_record=False, squeeze_me=True)
    if 'debug_data' in debug_data:
        data = debug_data['debug_data']
        print(f"Found {len(data)} entries in reduceAdaptive_debug.mat")
        
        # Look for entries that might match Step 2 Run 2
        # Since we don't have step/run info, look for entries with 5 generators
        matches = []
        for i, entry in enumerate(data):
            nrG = getattr(entry, 'nrG', None)
            if nrG == 5:
                dHmax = getattr(entry, 'dHmax', None)
                redIdx = getattr(entry, 'redIdx', None)
                last0Idx = getattr(entry, 'last0Idx', None)
                h_computed = getattr(entry, 'h_computed', None)
                
                matches.append({
                    'idx': i,
                    'nrG': nrG,
                    'dHmax': dHmax,
                    'redIdx': redIdx,
                    'last0Idx': last0Idx,
                    'h_computed': np.asarray(h_computed).flatten() if h_computed is not None else None
                })
        
        print(f"\nFound {len(matches)} entries with 5 generators")
        if matches:
            print("\nFirst matching entry:")
            m = matches[0]
            print(f"  nrG: {m['nrG']}")
            print(f"  dHmax: {m['dHmax']}")
            print(f"  redIdx: {m['redIdx']}")
            print(f"  last0Idx: {m['last0Idx']}")
            if m['h_computed'] is not None:
                h = m['h_computed']
                print(f"  h_computed: {h}")
                print(f"  h <= dHmax: {h <= m['dHmax']}")
                print(f"  Count <= dHmax: {np.sum(h <= m['dHmax'])}")
                
                # Compute expected final generators
                if m['redIdx'] and m['last0Idx'] is not None:
                    # Final = Gunred + diagonal
                    # Gunred = nrG - last0Idx - redIdx
                    # Diagonal = n (dimensions), but remove zeros
                    expected_unreduced = m['nrG'] - m['last0Idx'] - m['redIdx']
                    n_dims = len(h)  # h has length = nrG - last0Idx
                    # Diagonal has n_dims columns, but zeros are removed
                    # If Gred+Gzeros has zeros, they're removed in Python but not MATLAB
                    print(f"\n  Expected final generators:")
                    print(f"    Unreduced: {expected_unreduced}")
                    print(f"    Diagonal dimension: {n_dims}")
                    print(f"    Total (if no zeros): {expected_unreduced + n_dims}")
                    
except FileNotFoundError:
    print("[INFO] reduceAdaptive_debug.mat not found")
except Exception as e:
    print(f"[ERROR] Could not load reduction debug file: {e}")

print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)
print("Python reduces 5->2 generators, MATLAB reduces 5->4 generators.")
print("The difference is in how many generators are reduced (redIdx).")
print("This depends on:")
print("  1. dHmax value (computed from diagpercent and Gbox)")
print("  2. h array values (computed from gensdiag)")
print("  3. The comparison h <= dHmax")
print("\nTo fix this, we need to compare these values between Python and MATLAB.")
