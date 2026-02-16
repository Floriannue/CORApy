"""check_detailed_reduction - Check detailed reduction tracking data"""

import pickle
import numpy as np

# Load Python log
with open('upstream_python_log.pkl', 'rb') as f:
    data = pickle.load(f)

entries = data.get('upstreamLog', [])
print(f"Total entries: {len(entries)}")

# Find Step 3 entries with R tracking
step3_entries = [e for e in entries if e.get('step') == 3 and 'Rred_after_reduction' in e]

if step3_entries:
    entry = step3_entries[-1]
    Rred_after = entry.get('Rred_after_reduction', {})
    
    print(f"\nStep 3 - Detailed Reduction Tracking:")
    print(f"=" * 80)
    
    if 'reduction_details' in Rred_after:
        details = Rred_after['reduction_details']
        print(f"\nReduction Details:")
        print(f"  diagpercent: {details.get('diagpercent', 'N/A')}")
        print(f"  dHmax: {details.get('dHmax', 'N/A')}")
        print(f"  Gbox_sum: {details.get('Gbox_sum', 'N/A')}")
        
        if 'h_initial' in details and details['h_initial'] is not None:
            h_init = details['h_initial']
            print(f"\n  h_initial: {h_init}")
            print(f"    shape: {h_init.shape if hasattr(h_init, 'shape') else 'scalar'}")
            print(f"    max: {np.max(h_init) if hasattr(h_init, '__iter__') else h_init}")
            print(f"    min: {np.min(h_init) if hasattr(h_init, '__iter__') else h_init}")
        
        print(f"\n  hzeroIdx: {details.get('hzeroIdx', 'N/A')}")
        print(f"  last0Idx: {details.get('last0Idx', 'N/A')}")
        print(f"  gensred_shape: {details.get('gensred_shape', 'N/A')}")
        
        if 'h_computed' in details:
            h_comp = details['h_computed']
            print(f"\n  h_computed: {h_comp}")
            print(f"    shape: {h_comp.shape if hasattr(h_comp, 'shape') else 'scalar'}")
            print(f"    max: {np.max(h_comp) if hasattr(h_comp, '__iter__') else h_comp}")
            print(f"    min: {np.min(h_comp) if hasattr(h_comp, '__iter__') else h_comp}")
        
        print(f"\n  redIdx_arr: {details.get('redIdx_arr', 'N/A')}")
        print(f"  redIdx_0based: {details.get('redIdx_0based', 'N/A')}")
        print(f"  redIdx: {details.get('redIdx', 'N/A')}")
        print(f"  dHerror: {details.get('dHerror', 'N/A')}")
        print(f"  final_generators: {details.get('final_generators', 'N/A')}")
        
        if 'gredIdx' in details:
            gred = details['gredIdx']
            print(f"\n  gredIdx: {gred}")
            print(f"    length: {len(gred) if hasattr(gred, '__len__') else 'N/A'}")
    else:
        print("\nNo reduction_details found in Rred_after_reduction")
        print("This means detailed tracking was not enabled or captured")
    
    print(f"\n" + "=" * 80)
else:
    print("\nNo Step 3 entry with Rred_after_reduction found")
