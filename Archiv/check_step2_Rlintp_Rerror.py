"""Check Step 2's Rlintp_tracking and Rerror_tracking."""
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
    
    # Check Rlintp_tracking
    if hasattr(entry, 'Rlintp_tracking'):
        rlintp = entry.Rlintp_tracking
        if isinstance(rlintp, np.ndarray):
            if rlintp.size == 0:
                print(f"  Rlintp_tracking: empty array")
            else:
                rlintp_item = rlintp.item()
                if hasattr(rlintp_item, 'num_generators'):
                    num_gen = rlintp_item.num_generators
                    if isinstance(num_gen, np.ndarray):
                        num_gen = num_gen.item() if num_gen.size == 1 else num_gen
                    print(f"  Rlintp_tracking.num_generators: {num_gen}")
                else:
                    print(f"  Rlintp_tracking: {type(rlintp)} (has num_generators: {hasattr(rlintp_item, 'num_generators')})")
        else:
            # It's a mat_struct
            if hasattr(rlintp, 'num_generators'):
                num_gen = rlintp.num_generators
                if isinstance(num_gen, np.ndarray):
                    num_gen = num_gen.item() if num_gen.size == 1 else num_gen
                print(f"  Rlintp_tracking.num_generators: {num_gen}")
            else:
                print(f"  Rlintp_tracking: {type(rlintp)} (has num_generators: {hasattr(rlintp, 'num_generators')})")
    else:
        print(f"  Rlintp_tracking: NOT FOUND")
    
    # Check Rerror_tracking
    if hasattr(entry, 'Rerror_tracking'):
        rerror = entry.Rerror_tracking
        if isinstance(rerror, np.ndarray):
            if rerror.size == 0:
                print(f"  Rerror_tracking: empty array")
            else:
                rerror_item = rerror.item()
                if hasattr(rerror_item, 'num_generators'):
                    num_gen = rerror_item.num_generators
                    if isinstance(num_gen, np.ndarray):
                        num_gen = num_gen.item() if num_gen.size == 1 else num_gen
                    print(f"  Rerror_tracking.num_generators: {num_gen}")
                else:
                    print(f"  Rerror_tracking: {type(rerror)} (has num_generators: {hasattr(rerror_item, 'num_generators')})")
        else:
            # It's a mat_struct
            if hasattr(rerror, 'num_generators'):
                num_gen = rerror.num_generators
                if isinstance(num_gen, np.ndarray):
                    num_gen = num_gen.item() if num_gen.size == 1 else num_gen
                print(f"  Rerror_tracking.num_generators: {num_gen}")
            else:
                print(f"  Rerror_tracking: {type(rerror)} (has num_generators: {hasattr(rerror, 'num_generators')})")
    else:
        print(f"  Rerror_tracking: NOT FOUND")
    
    print()
