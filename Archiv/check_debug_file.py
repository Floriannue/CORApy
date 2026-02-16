"""
Check the structure of initReach_debug.pkl
"""
import pickle
import numpy as np

try:
    with open('initReach_debug.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Loaded file")
    print(f"Type: {type(data)}")
    print(f"Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    
    if isinstance(data, list) and len(data) > 0:
        print(f"\nFirst entry type: {type(data[0])}")
        print(f"First entry keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
        
        # Try to find Step 1 Run 1
        for i, entry in enumerate(data):
            if isinstance(entry, dict):
                step = entry.get('step')
                run = entry.get('run')
                print(f"\nEntry {i}: Step {step}, Run {run}")
                if step == 1 and run == 1:
                    print(f"  Found Step 1 Run 1!")
                    print(f"  Keys: {list(entry.keys())}")
                    for key in ['Rhom_tp_before_reduction', 'Rend_tp_after_reduction', 'diagpercent', 'dHmax', 'nrG', 'redIdx']:
                        if key in entry:
                            val = entry[key]
                            if isinstance(val, np.ndarray):
                                print(f"  {key}: shape {val.shape}, dtype {val.dtype}")
                            else:
                                print(f"  {key}: {val}")
                    break
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
