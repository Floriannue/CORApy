"""Debug MATLAB initReach_tracking access."""
import scipy.io
import numpy as np

# Load MATLAB log
matlab_log = scipy.io.loadmat('upstream_matlab_log.mat', squeeze_me=True, struct_as_record=False)
matlab_upstream_log = matlab_log['upstreamLog']

# Find Step 2 entries
matlab_step2_entries = []
for i, entry in enumerate(matlab_upstream_log):
    if hasattr(entry, 'step') and entry.step == 2:
        run = entry.run if hasattr(entry, 'run') else None
        if isinstance(run, np.ndarray):
            run = run.item() if run.size == 1 else run
        matlab_step2_entries.append((i, entry, run))

print(f"Found {len(matlab_step2_entries)} Step 2 entries")
print()

for idx, (i, entry, run) in enumerate(matlab_step2_entries):
    print(f"Step 2 Entry {idx+1} (index {i}, run={run}):")
    print(f"  Has initReach_tracking attribute: {hasattr(entry, 'initReach_tracking')}")
    
    if hasattr(entry, 'initReach_tracking'):
        initReach = entry.initReach_tracking
        print(f"  initReach_tracking type: {type(initReach)}")
        if isinstance(initReach, np.ndarray):
            print(f"  initReach_tracking shape: {initReach.shape}")
            print(f"  initReach_tracking size: {initReach.size}")
            if initReach.size > 0:
                print(f"  initReach_tracking is not empty")
                # Try to access nested fields
                if hasattr(initReach.item(), 'Rend_tp_num_generators'):
                    print(f"  Rend_tp_num_generators: {initReach.item().Rend_tp_num_generators}")
            else:
                print(f"  initReach_tracking is empty array")
        else:
            print(f"  initReach_tracking is not an array")
            if hasattr(initReach, 'Rend_tp_num_generators'):
                print(f"  Rend_tp_num_generators: {initReach.Rend_tp_num_generators}")
    print()
