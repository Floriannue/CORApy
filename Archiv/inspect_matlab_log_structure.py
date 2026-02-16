"""Inspect MATLAB log structure to understand how to access fields"""
import scipy.io as sio
import numpy as np

# Load MATLAB log
matlab_log_path = 'upstream_matlab_log.mat'
matlab_data = sio.loadmat(matlab_log_path, squeeze_me=False, struct_as_record=False)

# Get upstreamLog
upstream_log = matlab_data.get('upstreamLog', None)
if upstream_log is None:
    print("ERROR: upstreamLog not found in MATLAB log")
    exit(1)

print(f"upstreamLog type: {type(upstream_log)}")
print(f"upstreamLog shape: {upstream_log.shape if hasattr(upstream_log, 'shape') else 'N/A'}")
print(f"upstreamLog dtype: {upstream_log.dtype if hasattr(upstream_log, 'dtype') else 'N/A'}")
print()

# Check if it's a structured array
if hasattr(upstream_log, 'dtype') and upstream_log.dtype.names:
    print(f"Field names: {upstream_log.dtype.names}")
    print()
    
    # Get Step 3 entry (index 2)
    if upstream_log.size > 2:
        entry = upstream_log[2, 0]  # Step 3 is at index 2
        print(f"Entry 3 (Step {entry['step'][0,0] if 'step' in upstream_log.dtype.names else 'N/A'}):")
        print()
        
        # Check each field
        for field in ['Rmax_before_reduction', 'Rlinti_before_Rmax', 'RallError_before_Rmax', 'R_before_reduction']:
            if field in upstream_log.dtype.names:
                val = entry[field]
                print(f"{field}:")
                print(f"  Type: {type(val)}")
                print(f"  Shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
                print(f"  Size: {val.size if hasattr(val, 'size') else 'N/A'}")
                if val.size > 0:
                    if hasattr(val, 'dtype') and val.dtype.names:
                        print(f"  Is struct array: Yes")
                        print(f"  Struct fields: {val.dtype.names}")
                        # Try to access num_generators
                        if 'num_generators' in val.dtype.names:
                            num_gen = val['num_generators']
                            print(f"  num_generators: {num_gen[0,0] if num_gen.size > 0 else 'empty'}")
                    else:
                        print(f"  Is struct array: No")
                        print(f"  First few values: {val.flat[:min(5, val.size)]}")
                else:
                    print(f"  Empty")
                print()
            else:
                print(f"{field}: FIELD NOT FOUND")
                print()
else:
    print("Not a structured array - checking as object array")
    if upstream_log.size > 2:
        entry = upstream_log[2, 0]
        print(f"Entry type: {type(entry)}")
        if hasattr(entry, 'Rmax_before_reduction'):
            print(f"Rmax_before_reduction exists: {hasattr(entry, 'Rmax_before_reduction')}")
            rmax = entry.Rmax_before_reduction
            print(f"  Type: {type(rmax)}")
            print(f"  Shape: {rmax.shape if hasattr(rmax, 'shape') else 'N/A'}")
            if hasattr(rmax, 'num_generators'):
                print(f"  num_generators: {rmax.num_generators}")
