"""Check if Step 2's tracking fields are now populated after the fix."""
import scipy.io
import numpy as np

# Load MATLAB log
matlab_log = scipy.io.loadmat('upstream_matlab_log.mat', squeeze_me=True, struct_as_record=False)

# Get upstreamLog
upstream_log = matlab_log['upstreamLog']

# Find Step 2 entries
step2_entries = []
for i, entry in enumerate(upstream_log):
    if hasattr(entry, 'step') and entry.step == 2:
        step2_entries.append((i, entry))

print(f"Found {len(step2_entries)} Step 2 entries")
print()

for idx, (i, entry) in enumerate(step2_entries):
    print(f"Step 2 Entry {idx+1} (index {i}):")
    
    # Check timeStepequalHorizon_used
    if hasattr(entry, 'timeStepequalHorizon_used'):
        val = entry.timeStepequalHorizon_used
        if isinstance(val, np.ndarray):
            val = val.item() if val.size == 1 else val
        print(f"  timeStepequalHorizon_used: {val}")
    else:
        print(f"  timeStepequalHorizon_used: NOT FOUND")
    
    # Check Rtp_h_tracking
    if hasattr(entry, 'Rtp_h_tracking'):
        rtp_h = entry.Rtp_h_tracking
        if isinstance(rtp_h, np.ndarray) and rtp_h.size > 0:
            if hasattr(rtp_h.item(), 'num_generators'):
                num_gen = rtp_h.item().num_generators
                if isinstance(num_gen, np.ndarray):
                    num_gen = num_gen.item() if num_gen.size == 1 else num_gen
                print(f"  Rtp_h_tracking.num_generators: {num_gen}")
            else:
                print(f"  Rtp_h_tracking: {type(rtp_h)} (has num_generators: {hasattr(rtp_h.item(), 'num_generators')})")
        else:
            print(f"  Rtp_h_tracking: empty or not set")
    else:
        print(f"  Rtp_h_tracking: NOT FOUND")
    
    # Check Rerror_h_tracking
    if hasattr(entry, 'Rerror_h_tracking'):
        rerror_h = entry.Rerror_h_tracking
        if isinstance(rerror_h, np.ndarray) and rerror_h.size > 0:
            if hasattr(rerror_h.item(), 'num_generators'):
                num_gen = rerror_h.item().num_generators
                if isinstance(num_gen, np.ndarray):
                    num_gen = num_gen.item() if num_gen.size == 1 else num_gen
                print(f"  Rerror_h_tracking.num_generators: {num_gen}")
            else:
                print(f"  Rerror_h_tracking: {type(rerror_h)} (has num_generators: {hasattr(rerror_h.item(), 'num_generators')})")
        else:
            print(f"  Rerror_h_tracking: empty or not set")
    else:
        print(f"  Rerror_h_tracking: NOT FOUND")
    
    # Check Rtp_final_tracking
    if hasattr(entry, 'Rtp_final_tracking'):
        rtp_final = entry.Rtp_final_tracking
        if isinstance(rtp_final, np.ndarray) and rtp_final.size > 0:
            if hasattr(rtp_final.item(), 'num_generators'):
                num_gen = rtp_final.item().num_generators
                if isinstance(num_gen, np.ndarray):
                    num_gen = num_gen.item() if num_gen.size == 1 else num_gen
                print(f"  Rtp_final_tracking.num_generators: {num_gen}")
            else:
                print(f"  Rtp_final_tracking: {type(rtp_final)}")
        else:
            print(f"  Rtp_final_tracking: empty or not set")
    else:
        print(f"  Rtp_final_tracking: NOT FOUND")
    
    print()
