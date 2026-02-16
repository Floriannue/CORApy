"""Check what's in MATLAB initReach_tracking"""
import scipy.io
import numpy as np

print("=" * 80)
print("CHECKING MATLAB initReach_tracking STRUCTURE")
print("=" * 80)

matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

# Find Step 2 Run 2
for entry in matlab_log:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 2 and entry.run == 2:
            print(f"\nStep 2 Run 2 entry found")
            print(f"Has initReach_tracking: {hasattr(entry, 'initReach_tracking')}")
            if hasattr(entry, 'initReach_tracking'):
                it = entry.initReach_tracking
                print(f"initReach_tracking type: {type(it)}")
                if isinstance(it, np.ndarray):
                    print(f"initReach_tracking shape: {it.shape}")
                    if it.size > 0:
                        print(f"First element type: {type(it[0])}")
                        if hasattr(it[0], '_fieldnames'):
                            print(f"Fields: {it[0]._fieldnames}")
                            # Check for reduction_tp
                            if 'reduction_tp' in it[0]._fieldnames:
                                print(f"\nHas reduction_tp field!")
                                rt = it[0].reduction_tp
                                print(f"reduction_tp type: {type(rt)}")
                                if isinstance(rt, np.ndarray) and rt.size > 0:
                                    print(f"reduction_tp shape: {rt.shape}")
                                    if hasattr(rt[0], '_fieldnames'):
                                        print(f"reduction_tp fields: {rt[0]._fieldnames}")
                            else:
                                print(f"\nNo reduction_tp field")
                                print(f"Available fields: {it[0]._fieldnames}")
            break
