"""check_h_type - Check H type from tracking logs"""

import pickle
import numpy as np

# Load Python log
with open('upstream_python_log.pkl', 'rb') as f:
    data = pickle.load(f)
    log = data['upstreamLog']

# Find entry with H tracking
entries_with_h = [e for e in log if e.get('H_before_quadmap')]
print(f"Entries with H_before_quadmap: {len(entries_with_h)}")

if entries_with_h:
    entry = entries_with_h[0]
    H = entry['H_before_quadmap']
    step = entry.get('step', 'unknown')
    
    print(f"\nStep {step}:")
    print(f"H type: {type(H)}")
    print(f"H length: {len(H) if H else 0}")
    
    if H and len(H) > 0:
        H0 = H[0]
        print(f"\nH[0] type: {type(H0)}")
        
        if isinstance(H0, dict):
            print(f"H[0] keys: {list(H0.keys())}")
            if 'inf' in H0 and 'sup' in H0:
                print("H[0] is Interval (has inf and sup)")
                print(f"  inf max: {np.max(np.abs(H0['inf'])) if H0['inf'] is not None else 'None'}")
                print(f"  sup max: {np.max(np.abs(H0['sup'])) if H0['sup'] is not None else 'None'}")
            elif 'matrix' in H0:
                print("H[0] is numeric (has matrix)")
                print(f"  matrix max: {np.max(np.abs(H0['matrix'])) if H0['matrix'] is not None else 'None'}")
        else:
            print(f"H[0] is not a dict: {H0}")
