"""Debug Step 2 tracking to see what's actually in the log."""
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
    print(f"  All fields: {dir(entry)}")
    print()
    
    # Check all tracking-related fields
    tracking_fields = ['timeStepequalHorizon_used', 'Rtp_h_tracking', 'Rerror_h_tracking', 
                       'Rtp_final_tracking', 'Rlintp_tracking', 'Rerror_tracking']
    
    for field in tracking_fields:
        if hasattr(entry, field):
            val = getattr(entry, field)
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    print(f"  {field}: empty array")
                else:
                    print(f"  {field}: {type(val)}, size={val.size}, shape={val.shape}")
                    if val.size == 1:
                        try:
                            item = val.item()
                            if hasattr(item, '__dict__'):
                                print(f"    -> struct with fields: {dir(item)}")
                            else:
                                print(f"    -> {type(item)}: {item}")
                        except:
                            print(f"    -> could not get item")
            else:
                print(f"  {field}: {type(val)} = {val}")
        else:
            print(f"  {field}: NOT FOUND")
    print()
