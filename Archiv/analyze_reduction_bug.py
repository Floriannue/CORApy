"""Analyze the reduction bug - why Python reduces 3 when h > dHmax"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("ANALYZING REDUCTION BUG")
print("=" * 80)

# Load Python reduction debug file
try:
    with open('reduceAdaptive_debug_python.pkl', 'rb') as f:
        python_debug = pickle.load(f)
    print(f"[OK] Found {len(python_debug)} entries in Python debug file")
    
    # Find entry with 5 generators (Step 2 Run 2)
    for i, entry in enumerate(python_debug):
        if isinstance(entry, dict) and entry.get('nrG') == 5:
            print(f"\n[FOUND] Entry {i} with 5 generators:")
            print(f"  dHmax: {entry.get('dHmax')}")
            print(f"  last0Idx: {entry.get('last0Idx')}")
            print(f"  redIdx: {entry.get('redIdx')}")
            print(f"  redIdx_0based: {entry.get('redIdx_0based')}")
            h_computed = entry.get('h_computed')
            if h_computed is not None:
                h = np.asarray(h_computed)
                dHmax = entry.get('dHmax')
                print(f"  h_computed: {h}")
                print(f"  h <= dHmax: {h <= dHmax}")
                print(f"  Count <= dHmax: {np.sum(h <= dHmax)}")
                
                # Analyze the logic
                print(f"\n  ANALYSIS:")
                if len(h) > 0:
                    redIdx_arr = np.where(h <= dHmax)[0]
                    print(f"    redIdx_arr (0-based): {redIdx_arr}")
                    if len(redIdx_arr) > 0:
                        expected_redIdx_0based = redIdx_arr[-1]
                        expected_redIdx = expected_redIdx_0based + 1
                        print(f"    Expected redIdx_0based: {expected_redIdx_0based}")
                        print(f"    Expected redIdx (1-based): {expected_redIdx}")
                        print(f"    Actual redIdx: {entry.get('redIdx')}")
                        if entry.get('redIdx') != expected_redIdx:
                            print(f"    [BUG] redIdx mismatch!")
                    else:
                        print(f"    Expected redIdx: 0 (no generators satisfy h <= dHmax)")
                        print(f"    Actual redIdx: {entry.get('redIdx')}")
                        if entry.get('redIdx') != 0:
                            print(f"    [BUG] redIdx should be 0 but is {entry.get('redIdx')}!")
            
            # Check final generators
            final_gens = entry.get('final_generators')
            if final_gens:
                print(f"\n  Final generators: {final_gens}")
                print(f"  Expected: 5 - {entry.get('last0Idx')} - {entry.get('redIdx')} unreduced + diagonal")
            break
except FileNotFoundError:
    print("[INFO] reduceAdaptive_debug_python.pkl not found")

# Load MATLAB reduction debug file
print("\n" + "=" * 80)
print("MATLAB REDUCTION DEBUG")
print("=" * 80)

try:
    matlab_data = scipy.io.loadmat('reduceAdaptive_debug.mat', struct_as_record=False, squeeze_me=True)
    if 'debug_data' in matlab_data:
        data = matlab_data['debug_data']
        print(f"[OK] Found {len(data)} entries in MATLAB debug file")
        
        # Find entries with 5 generators
        matches = []
        for i, entry in enumerate(data):
            nrG = getattr(entry, 'nrG', None)
            if nrG == 5:
                matches.append((i, entry))
        
        print(f"\n[FOUND] {len(matches)} entries with 5 generators")
        if matches:
            idx, entry = matches[0]
            print(f"\nEntry {idx}:")
            dHmax = getattr(entry, 'dHmax', None)
            last0Idx = getattr(entry, 'last0Idx', None)
            redIdx = getattr(entry, 'redIdx', None)
            h_computed = getattr(entry, 'h_computed', None)
            
            print(f"  dHmax: {dHmax}")
            print(f"  last0Idx: {last0Idx}")
            print(f"  redIdx: {redIdx}")
            
            if h_computed is not None:
                h = np.asarray(h_computed).flatten()
                print(f"  h_computed: {h}")
                print(f"  h <= dHmax: {h <= dHmax}")
                print(f"  Count <= dHmax: {np.sum(h <= dHmax)}")
                
                # Analyze
                if len(h) > 0:
                    redIdx_arr = np.where(h <= dHmax)[0]
                    print(f"  redIdx_arr (0-based): {redIdx_arr}")
                    if len(redIdx_arr) > 0:
                        expected_redIdx = redIdx_arr[-1] + 1
                        print(f"  Expected redIdx (1-based): {expected_redIdx}")
                        print(f"  Actual redIdx: {redIdx}")
                    else:
                        print(f"  Expected redIdx: 0")
                        print(f"  Actual redIdx: {redIdx}")
except FileNotFoundError:
    print("[INFO] reduceAdaptive_debug.mat not found")
except Exception as e:
    print(f"[ERROR] Could not load MATLAB debug file: {e}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print("Compare the redIdx values and h arrays between Python and MATLAB")
print("to find where the divergence occurs.")
