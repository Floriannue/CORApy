"""Check MATLAB log structure"""
import scipy.io as sio
import numpy as np

# Try both loading methods
print("=" * 80)
print("CHECKING MATLAB LOG STRUCTURE")
print("=" * 80)

# Method 1: struct_as_record=False
print("\nMethod 1: struct_as_record=False")
data1 = sio.loadmat('upstream_matlab_log.mat', squeeze_me=False, struct_as_record=False)
log1 = data1.get('upstreamLog', [])
print(f"  Type: {type(log1)}")
if isinstance(log1, np.ndarray):
    print(f"  Shape: {log1.shape}")
    if log1.size > 0:
        e = log1[1, 0]  # Step 2 entry
        print(f"  Entry type: {type(e)}")
        print(f"  Step: {getattr(e, 'step', 'N/A')}")
        print(f"  Run: {getattr(e, 'run', 'N/A')}")
        # Check for tracking fields
        print(f"  Has Rtp_final_tracking: {hasattr(e, 'Rtp_final_tracking')}")
        print(f"  Has Rlintp_tracking: {hasattr(e, 'Rlintp_tracking')}")
        print(f"  Has Rerror_tracking: {hasattr(e, 'Rerror_tracking')}")
        print(f"  Has Rtp_h_tracking: {hasattr(e, 'Rtp_h_tracking')}")
        if hasattr(e, 'Rtp_final_tracking'):
            rtp_final = e.Rtp_final_tracking
            print(f"    Rtp_final type: {type(rtp_final)}")
            if hasattr(rtp_final, 'num_generators'):
                print(f"    Rtp_final num_generators: {rtp_final.num_generators}")
            if hasattr(rtp_final, 'timeStepequalHorizon_used'):
                print(f"    Rtp_final timeStepequalHorizon_used: {rtp_final.timeStepequalHorizon_used}")

# Method 2: struct_as_record=True
print("\nMethod 2: struct_as_record=True")
data2 = sio.loadmat('upstream_matlab_log.mat', squeeze_me=True, struct_as_record=True)
log2 = data2.get('upstreamLog', [])
print(f"  Type: {type(log2)}")
if isinstance(log2, np.ndarray) and log2.size > 0:
    e = log2[1]  # Step 2 entry
    print(f"  Entry type: {type(e)}")
    if hasattr(e, '__dict__'):
        print(f"  Fields: {list(e.__dict__.keys())[:15]}")
    elif hasattr(e, '_fieldnames'):
        print(f"  Fields: {list(e._fieldnames)[:15]}")
    else:
        print(f"  Dir: {[x for x in dir(e) if not x.startswith('_')][:15]}")
