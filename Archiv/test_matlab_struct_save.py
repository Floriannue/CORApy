"""Test if MATLAB struct fields are saved correctly"""
import scipy.io as sio
import numpy as np

# Load MATLAB log
matlab_log_path = 'upstream_matlab_log.mat'
matlab_data = sio.loadmat(matlab_log_path, squeeze_me=False, struct_as_record=False)

# Get upstreamLog
upstream_log = matlab_data.get('upstreamLog', None)
if upstream_log is None:
    print("ERROR: upstreamLog not found")
    exit(1)

print(f"upstreamLog type: {type(upstream_log)}")
print(f"upstreamLog shape: {upstream_log.shape}")
print()

# Check Step 3 entry (index 2)
if upstream_log.size > 2:
    entry = upstream_log[2, 0]  # Step 3 is at index 2
    print(f"Entry 3 type: {type(entry)}")
    print(f"Entry 3 attributes: {dir(entry)}")
    print()
    
    # Try to access Rmax_before_reduction
    if hasattr(entry, 'Rmax_before_reduction'):
        rmax = entry.Rmax_before_reduction
        print(f"Rmax_before_reduction type: {type(rmax)}")
        print(f"Rmax_before_reduction shape: {rmax.shape if hasattr(rmax, 'shape') else 'N/A'}")
        print(f"Rmax_before_reduction size: {rmax.size if hasattr(rmax, 'size') else 'N/A'}")
        
        # Try to access as struct
        if isinstance(rmax, np.ndarray) and rmax.size > 0:
            if hasattr(rmax, 'dtype') and rmax.dtype.names:
                print(f"Rmax is structured array with fields: {rmax.dtype.names}")
                if 'num_generators' in rmax.dtype.names:
                    num_gen = rmax['num_generators']
                    print(f"num_generators value: {num_gen}")
                    if num_gen.size > 0:
                        print(f"num_generators[0,0]: {num_gen[0,0]}")
            else:
                # Try accessing as object
                if rmax.size == 1:
                    rmax_obj = rmax.item() if hasattr(rmax, 'item') else rmax[0,0]
                    print(f"Rmax object type: {type(rmax_obj)}")
                    if hasattr(rmax_obj, 'num_generators'):
                        print(f"num_generators: {rmax_obj.num_generators}")
                    print(f"Rmax object attributes: {dir(rmax_obj)}")
        else:
            print("Rmax is empty or not a structured array")
    else:
        print("Rmax_before_reduction attribute not found")
