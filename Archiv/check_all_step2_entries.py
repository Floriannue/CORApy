"""Check all entries to see Step 2's run numbers."""
import scipy.io
import numpy as np

# Load MATLAB log
matlab_log = scipy.io.loadmat('upstream_matlab_log.mat', squeeze_me=True, struct_as_record=False)

# Get upstreamLog
upstream_log = matlab_log['upstreamLog']

# Find all Step 2 entries
step2_entries = []
for i, entry in enumerate(upstream_log):
    if hasattr(entry, 'step'):
        step = entry.step
        if isinstance(step, np.ndarray):
            step = step.item() if step.size == 1 else step
        if step == 2:
            run = entry.run if hasattr(entry, 'run') else None
            if isinstance(run, np.ndarray):
                run = run.item() if run.size == 1 else run
            step2_entries.append((i, entry, run))

print(f"Found {len(step2_entries)} Step 2 entries")
print()

for idx, (i, entry, run) in enumerate(step2_entries):
    print(f"Step 2 Entry {idx+1} (index {i}, run {run}):")
    print(f"  Has timeStepequalHorizon_used: {hasattr(entry, 'timeStepequalHorizon_used')}")
    print(f"  Has Rtp_h_tracking: {hasattr(entry, 'Rtp_h_tracking')}")
    print(f"  Has Rerror_h_tracking: {hasattr(entry, 'Rerror_h_tracking')}")
    print()
