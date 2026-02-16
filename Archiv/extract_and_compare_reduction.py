"""Extract and compare reduction parameters from available data"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("EXTRACTING AND COMPARING REDUCTION PARAMETERS")
print("=" * 80)

# Python Step 2 Run 2
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        if 'initReach_tracking' in entry:
            py_entry = entry['initReach_tracking']
            break

if py_entry is None:
    print("[ERROR] Could not find Python Step 2 Run 2")
    exit(1)

print("[OK] Found Python Step 2 Run 2")
print(f"  Rhom_tp_num_generators: {py_entry.get('Rhom_tp_num_generators', 'N/A')}")
print(f"  Rend_tp_num_generators: {py_entry.get('Rend_tp_num_generators', 'N/A')}")
print(f"  redFactor: {py_entry.get('redFactor', 'N/A')}")

# Try to find matching MATLAB reduction from debug file
print("\n" + "-" * 80)
print("SEARCHING MATLAB REDUCTION DEBUG FILE")
print("-" * 80)

try:
    debug_data = scipy.io.loadmat('reduceAdaptive_debug.mat', struct_as_record=False, squeeze_me=True)
    if 'debug_data' in debug_data:
        data = debug_data['debug_data']
        print(f"Found {len(data)} reduction debug entries")
        
        # Look for entries that match the scenario:
        # - 5 generators input (Rhom_tp_num_generators)
        # - Similar dHmax (computed from redFactor)
        py_rhom_tp = py_entry.get('Rhom_tp_num_generators')
        py_redFactor = py_entry.get('redFactor')
        
        print(f"\nLooking for entries matching:")
        print(f"  Input generators: {py_rhom_tp}")
        print(f"  redFactor: {py_redFactor}")
        
        # Compute expected dHmax (approximate)
        # dHmax = (diagpercent * 2) * sqrt(sum(Gbox^2))
        # diagpercent = sqrt(redFactor)
        if py_redFactor:
            diagpercent = np.sqrt(py_redFactor)
            print(f"  Expected diagpercent: {diagpercent}")
        
        # Search for matching entries
        matches = []
        for i, entry in enumerate(data):
            nrG = getattr(entry, 'nrG', None)
            diagpercent_entry = getattr(entry, 'diagpercent', None)
            if nrG == py_rhom_tp:
                matches.append((i, entry, nrG, diagpercent_entry))
        
        print(f"\nFound {len(matches)} entries with {py_rhom_tp} generators")
        
        # Show first few matches
        for idx, entry, nrG, diagpercent in matches[:5]:
            dHmax = getattr(entry, 'dHmax', None)
            h_computed = getattr(entry, 'h_computed', None)
            redIdx = getattr(entry, 'redIdx', None)
            last0Idx = getattr(entry, 'last0Idx', None)
            
            print(f"\n  Entry {idx}:")
            print(f"    diagpercent: {diagpercent}")
            print(f"    dHmax: {dHmax}")
            print(f"    last0Idx: {last0Idx}")
            if h_computed is not None:
                h_arr = np.asarray(h_computed).flatten()
                print(f"    h_computed: {h_arr}")
                print(f"    h_computed length: {len(h_arr)}")
                if dHmax:
                    h_le_dHmax = h_arr <= dHmax
                    print(f"    h <= dHmax: {h_le_dHmax}")
                    print(f"    Count <= dHmax: {np.sum(h_le_dHmax)}")
            print(f"    redIdx: {redIdx}")
            if redIdx and last0Idx is not None:
                final_gens = nrG - last0Idx - redIdx + 2  # Approximate: unreduced + diagonal
                print(f"    Estimated final generators: {final_gens}")
                
                # Check if this matches MATLAB's result (4 generators)
                if final_gens == 4:
                    print(f"    [POTENTIAL MATCH] This could be the Step 2 Run 2 reduction!")
                    
except FileNotFoundError:
    print("[ERROR] reduceAdaptive_debug.mat not found")
except Exception as e:
    print(f"[ERROR] Could not process reduction debug file: {e}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("Python Step 2 Run 2:")
print(f"  Input: {py_entry.get('Rhom_tp_num_generators', 'N/A')} generators")
print(f"  Output: {py_entry.get('Rend_tp_num_generators', 'N/A')} generators")
print(f"  Reduced: {py_entry.get('Rhom_tp_num_generators', 0) - py_entry.get('Rend_tp_num_generators', 0)} generators")
print("\nMATLAB should produce:")
print(f"  Input: {py_entry.get('Rhom_tp_num_generators', 'N/A')} generators (should match)")
print(f"  Output: 4 generators (expected)")
print(f"  Reduced: {py_entry.get('Rhom_tp_num_generators', 0) - 4} generators (expected)")
