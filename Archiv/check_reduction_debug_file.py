"""Check if reduction parameters are in reduceAdaptive_debug.mat"""
import scipy.io
import numpy as np

print("=" * 80)
print("CHECKING REDUCTION DEBUG FILE")
print("=" * 80)

try:
    debug_data = scipy.io.loadmat('reduceAdaptive_debug.mat', struct_as_record=False, squeeze_me=True)
    if 'debug_data' in debug_data:
        data = debug_data['debug_data']
        print(f"[OK] Found reduceAdaptive_debug.mat with {len(data)} entries")
        
        # Find entries for Step 2 Run 2 or Step 1 Run 1
        print("\nSearching for relevant entries...")
        for i, entry in enumerate(data):
            step = getattr(entry, 'step', None)
            run = getattr(entry, 'run', None)
            if step == 2 and run == 2:
                print(f"\nFound Step 2 Run 2 entry at index {i}:")
                print(f"  dHmax: {getattr(entry, 'dHmax', 'N/A')}")
                print(f"  nrG: {getattr(entry, 'nrG', 'N/A')}")
                print(f"  last0Idx: {getattr(entry, 'last0Idx', 'N/A')}")
                h_computed = getattr(entry, 'h_computed', None)
                if h_computed is not None:
                    h_arr = np.asarray(h_computed).flatten()
                    print(f"  h_computed: {h_arr}")
                    print(f"  h_computed length: {len(h_arr)}")
                redIdx = getattr(entry, 'redIdx', None)
                print(f"  redIdx: {redIdx}")
                break
            elif step == 1 and run == 1:
                print(f"\nFound Step 1 Run 1 entry at index {i}:")
                print(f"  dHmax: {getattr(entry, 'dHmax', 'N/A')}")
                print(f"  nrG: {getattr(entry, 'nrG', 'N/A')}")
                print(f"  last0Idx: {getattr(entry, 'last0Idx', 'N/A')}")
                h_computed = getattr(entry, 'h_computed', None)
                if h_computed is not None:
                    h_arr = np.asarray(h_computed).flatten()
                    print(f"  h_computed: {h_arr}")
                    print(f"  h_computed length: {len(h_arr)}")
                redIdx = getattr(entry, 'redIdx', None)
                print(f"  redIdx: {redIdx}")
except FileNotFoundError:
    print("[ERROR] reduceAdaptive_debug.mat not found")
    print("        This file is created by priv_reduceAdaptive.m during reduction")
except Exception as e:
    print(f"[ERROR] Could not load reduceAdaptive_debug.mat: {e}")
