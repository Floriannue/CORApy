"""
Analyze reduction details for Step 1 Run 1 to find divergence
"""
import pickle
import numpy as np

try:
    with open('initReach_debug.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"[OK] Loaded debug file with {len(data)} entries\n")
    
    # Find Step 1 Run 1
    step1_run1 = None
    for entry in data:
        if entry.get('step') == 1 and entry.get('run') == 1:
            step1_run1 = entry
            break
    
    if step1_run1 is None:
        print("[ERROR] Step 1 Run 1 not found")
        print("Available entries:")
        for entry in data:
            print(f"  Step {entry.get('step')}, Run {entry.get('run')}")
        exit(1)
    
    print("="*80)
    print("Step 1 Run 1 Reduction Analysis")
    print("="*80)
    
    # Rhom_tp before reduction
    print(f"\nRhom_tp (before reduction):")
    print(f"  Generators: {step1_run1.get('Rhom_tp_num_generators', 'N/A')}")
    
    # Rend.tp after reduction
    print(f"\nRend.tp (after reduction):")
    print(f"  Generators: {step1_run1.get('Rend_tp_num_generators', 'N/A')}")
    
    # Reduction parameters for Rend.tp
    print(f"\nReduction parameters for Rend.tp:")
    if 'reduction_tp_diagpercent' in step1_run1:
        print(f"  diagpercent: {step1_run1.get('reduction_tp_diagpercent')}")
        print(f"  dHmax: {step1_run1.get('reduction_tp_dHmax')}")
        print(f"  nrG (initial generators): {step1_run1.get('reduction_tp_nrG')}")
        print(f"  last0Idx: {step1_run1.get('reduction_tp_last0Idx')}")
        print(f"  redIdx (1-based): {step1_run1.get('reduction_tp_redIdx')}")
        print(f"  redIdx_0based (0-based): {step1_run1.get('reduction_tp_redIdx_0based')}")
        print(f"  dHerror: {step1_run1.get('reduction_tp_dHerror')}")
        print(f"  gredIdx_len: {step1_run1.get('reduction_tp_gredIdx_len')}")
        
        # h array
        h_computed = step1_run1.get('reduction_tp_h_computed')
        if h_computed is not None:
            h_arr = np.asarray(h_computed)
            print(f"\n  h array (shape {h_arr.shape}):")
            print(f"    Values: {h_arr}")
            print(f"    Min: {np.min(h_arr)}, Max: {np.max(h_arr)}")
            
            # Check how many are <= dHmax
            dHmax = step1_run1.get('reduction_tp_dHmax')
            if dHmax is not None:
                h_le_dHmax = h_arr <= dHmax
                count = np.sum(h_le_dHmax)
                print(f"    Values <= dHmax ({dHmax:.6e}): {count} out of {len(h_arr)}")
                print(f"    Indices where h <= dHmax: {np.where(h_le_dHmax)[0]}")
        
        # gredIdx
        gredIdx = step1_run1.get('reduction_tp_gredIdx')
        if gredIdx is not None:
            gredIdx_arr = np.asarray(gredIdx)
            print(f"\n  gredIdx (shape {gredIdx_arr.shape}):")
            print(f"    Values: {gredIdx_arr}")
    else:
        print("  [ERROR] Reduction details not found")
    
    print("\n" + "="*80)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
