"""Compare Step 2 Run 2 reduction values in detail"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING STEP 2 RUN 2 REDUCTION VALUES")
print("=" * 80)

# Load Python tracking
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

# Get Rhom_tp data
if 'initReach_tracking' in py_entry:
    it = py_entry['initReach_tracking']
    rhom_tp_gens = np.asarray(it.get('Rhom_tp_generators'))
    redFactor = it.get('redFactor', 0.0005)
    
    print(f"\nPython Rhom_tp:")
    print(f"  Generator matrix shape: {rhom_tp_gens.shape}")
    print(f"  redFactor: {redFactor}")
    
    # Compute dHmax
    diagpercent = np.sqrt(redFactor)
    Gabs = np.abs(rhom_tp_gens)
    Gbox = np.sum(Gabs, axis=1, keepdims=True)
    dHmax_py = (diagpercent * 2) * np.sqrt(np.sum(Gbox ** 2))
    print(f"  Computed dHmax: {dHmax_py:.12e}")
    print(f"  diagpercent: {diagpercent:.12e}")
    print(f"  Gbox: {Gbox.flatten()}")

# Load MATLAB reduction debug file
print("\n" + "=" * 80)
print("MATLAB REDUCTION DEBUG FILE")
print("=" * 80)

try:
    matlab_data = scipy.io.loadmat('reduceAdaptive_debug.mat', struct_as_record=False, squeeze_me=True)
    if 'debug_data' in matlab_data:
        data = matlab_data['debug_data']
        print(f"[OK] Found {len(data)} entries in MATLAB debug file")
        
        # Find entries with 5 generators and matching dHmax
        matches = []
        for i, entry in enumerate(data):
            nrG = getattr(entry, 'nrG', None)
            dHmax = getattr(entry, 'dHmax', None)
            if nrG == 5 and dHmax is not None:
                # Check if dHmax is close to Python's
                if abs(dHmax - dHmax_py) < 1e-6:
                    matches.append((i, entry))
        
        print(f"\n[FOUND] {len(matches)} entries with 5 generators and matching dHmax")
        
        if matches:
            idx, entry = matches[0]
            print(f"\nMATLAB Entry {idx}:")
            dHmax_mat = getattr(entry, 'dHmax', None)
            last0Idx = getattr(entry, 'last0Idx', None)
            redIdx = getattr(entry, 'redIdx', None)
            redIdx_0based = getattr(entry, 'redIdx_0based', None)
            h_computed = getattr(entry, 'h_computed', None)
            h_initial = getattr(entry, 'h_initial', None)
            nrG = getattr(entry, 'nrG', None)
            
            print(f"  dHmax: {dHmax_mat:.12e}")
            print(f"  nrG: {nrG}")
            print(f"  last0Idx: {last0Idx}")
            print(f"  redIdx: {redIdx}")
            print(f"  redIdx_0based: {redIdx_0based}")
            
            if h_computed is not None:
                h = np.asarray(h_computed).flatten()
                print(f"  h_computed: {h}")
                print(f"  h <= dHmax: {h <= dHmax_mat}")
                print(f"  Count <= dHmax: {np.sum(h <= dHmax_mat)}")
                
                # Compute expected redIdx
                redIdx_arr = np.where(h <= dHmax_mat)[0]
                print(f"  redIdx_arr (0-based): {redIdx_arr}")
                if len(redIdx_arr) > 0:
                    expected_redIdx = redIdx_arr[-1] + 1
                    print(f"  Expected redIdx (1-based): {expected_redIdx}")
                    print(f"  Actual redIdx: {redIdx}")
                else:
                    print(f"  Expected redIdx: 0")
                    print(f"  Actual redIdx: {redIdx}")
            
            if h_initial is not None:
                h_init = np.asarray(h_initial).flatten()
                print(f"\n  h_initial: {h_init}")
                print(f"  h_initial <= dHmax: {h_init <= dHmax_mat}")
                print(f"  Count <= dHmax: {np.sum(h_init <= dHmax_mat)}")
        
        # Also check for entries with different dHmax (might be from different step/run)
        print(f"\n" + "=" * 80)
        print("ALL ENTRIES WITH 5 GENERATORS")
        print("=" * 80)
        
        all_5gen = []
        for i, entry in enumerate(data):
            nrG = getattr(entry, 'nrG', None)
            if nrG == 5:
                dHmax = getattr(entry, 'dHmax', None)
                redIdx = getattr(entry, 'redIdx', None)
                h_computed = getattr(entry, 'h_computed', None)
                all_5gen.append((i, entry, dHmax, redIdx, h_computed))
        
        print(f"Found {len(all_5gen)} total entries with 5 generators")
        for i, (idx, entry, dHmax, redIdx, h_computed) in enumerate(all_5gen[:5]):  # Show first 5
            print(f"\n  Entry {idx}:")
            if dHmax is not None:
                print(f"    dHmax: {dHmax:.12e}")
            else:
                print(f"    dHmax: None")
            print(f"    redIdx: {redIdx}")
            if h_computed is not None and dHmax is not None:
                h = np.asarray(h_computed).flatten()
                print(f"    h_computed: {h}")
                print(f"    h <= dHmax: {np.sum(h <= dHmax)}")
                
except FileNotFoundError:
    print("[INFO] reduceAdaptive_debug.mat not found")
except Exception as e:
    print(f"[ERROR] Could not load MATLAB debug file: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("1. Compare dHmax values between Python and MATLAB")
print("2. Compare h_computed arrays")
print("3. Verify redIdx computation logic")
print("4. Check if the issue is in gensdiag computation or h computation")
