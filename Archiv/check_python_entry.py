"""Check Python entry structure"""
import pickle

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 1 Run 1
for entry in python_upstream:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 1 and entry.run == 1:
            print(f"Found Step 1 Run 1 (object)")
            print(f"Type: {type(entry)}")
            print(f"Has initReach_tracking: {hasattr(entry, 'initReach_tracking')}")
            if hasattr(entry, 'initReach_tracking'):
                it = entry.initReach_tracking
                print(f"initReach_tracking type: {type(it)}")
                if isinstance(it, dict):
                    print(f"Keys: {list(it.keys())}")
                    if 'reduction_tp' in it:
                        print(f"Has reduction_tp: True")
                        print(f"reduction_tp type: {type(it['reduction_tp'])}")
                        if isinstance(it['reduction_tp'], dict):
                            print(f"reduction_tp keys: {list(it['reduction_tp'].keys())}")
            break
    elif isinstance(entry, dict):
        if entry.get('step') == 1 and entry.get('run') == 1:
            print(f"Found Step 1 Run 1 (dict)")
            print(f"Keys: {list(entry.keys())}")
            if 'initReach_tracking' in entry:
                it = entry['initReach_tracking']
                print(f"initReach_tracking type: {type(it)}")
                if isinstance(it, dict):
                    print(f"Keys: {list(it.keys())}")
            break
