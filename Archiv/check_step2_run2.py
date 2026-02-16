"""Check Step 2 Run 2 entries"""
import pickle
import scipy.io
import numpy as np

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        print("Python Step 2 Run 2:")
        print(f"  Keys: {list(entry.keys())}")
        if 'initReach_tracking' in entry:
            it = entry['initReach_tracking']
            print(f"  initReach_tracking type: {type(it)}")
            if isinstance(it, dict):
                print(f"  initReach_tracking keys: {list(it.keys())}")
                if 'reduction_tp' in it:
                    print(f"  Has reduction_tp: True")
                    rt = it['reduction_tp']
                    print(f"  reduction_tp type: {type(rt)}")
                    if isinstance(rt, dict):
                        print(f"  reduction_tp keys: {list(rt.keys())}")
        break

# MATLAB
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

for entry in matlab_log:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 2 and entry.run == 2:
            print("\nMATLAB Step 2 Run 2:")
            print(f"  Has initReach_tracking: {hasattr(entry, 'initReach_tracking')}")
            if hasattr(entry, 'initReach_tracking'):
                it = entry.initReach_tracking
                print(f"  initReach_tracking type: {type(it)}")
                if isinstance(it, np.ndarray):
                    print(f"  initReach_tracking shape: {it.shape}")
                    if it.size > 0:
                        print(f"  First element type: {type(it[0])}")
                        if hasattr(it[0], 'reduction_tp'):
                            print(f"  Has reduction_tp: True")
                            rt = it[0].reduction_tp
                            print(f"  reduction_tp type: {type(rt)}")
                            if hasattr(rt, '_fieldnames'):
                                print(f"  reduction_tp fields: {rt._fieldnames}")
            break
