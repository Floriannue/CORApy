"""Check Step 2 entries and their run numbers."""
import scipy.io
import numpy as np

# Load MATLAB log
matlab_log = scipy.io.loadmat('upstream_matlab_log.mat', squeeze_me=True, struct_as_record=False)

# Get upstreamLog
upstream_log = matlab_log['upstreamLog']

# Find all Step 2 entries
step2_entries = []
for i, entry in enumerate(upstream_log):
    if hasattr(entry, 'step') and entry.step == 2:
        run = entry.run if hasattr(entry, 'run') else None
        if isinstance(run, np.ndarray):
            run = run.item() if run.size == 1 else run
        step2_entries.append((i, entry, run))

print(f"Found {len(step2_entries)} Step 2 entries")
print()

for idx, (i, entry, run) in enumerate(step2_entries):
    print(f"Step 2 Entry {idx+1} (index {i}, run {run}):")
    
    # Check timeStepequalHorizon_used
    if hasattr(entry, 'timeStepequalHorizon_used'):
        val = entry.timeStepequalHorizon_used
        if isinstance(val, np.ndarray):
            if val.size == 0:
                print(f"  timeStepequalHorizon_used: empty array")
            else:
                val = val.item() if val.size == 1 else val
                print(f"  timeStepequalHorizon_used: {val}")
        else:
            print(f"  timeStepequalHorizon_used: {val}")
    else:
        print(f"  timeStepequalHorizon_used: NOT FOUND")
    
    # Check Rtp_h_tracking
    if hasattr(entry, 'Rtp_h_tracking'):
        rtp_h = entry.Rtp_h_tracking
        if isinstance(rtp_h, np.ndarray):
            if rtp_h.size == 0:
                print(f"  Rtp_h_tracking: empty array")
            else:
                if hasattr(rtp_h.item(), 'num_generators'):
                    num_gen = rtp_h.item().num_generators
                    if isinstance(num_gen, np.ndarray):
                        num_gen = num_gen.item() if num_gen.size == 1 else num_gen
                    print(f"  Rtp_h_tracking.num_generators: {num_gen}")
                else:
                    print(f"  Rtp_h_tracking: {type(rtp_h)} (has num_generators: {hasattr(rtp_h.item(), 'num_generators')})")
        else:
            print(f"  Rtp_h_tracking: {type(rtp_h)}")
    else:
        print(f"  Rtp_h_tracking: NOT FOUND")
    
    print()
