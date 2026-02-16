"""Check if Rmax, Rlinti, and RallError tracking is captured in MATLAB log"""
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

print(f"Total entries in upstreamLog: {len(upstream_log)}")
print()

# Check Step 3 (index 2)
step_idx = 2
if step_idx < upstream_log.shape[0]:
    entry = upstream_log[step_idx, 0]  # Use [idx, 0] for 2D array
    print(f"Entry {step_idx + 1} (Step {entry.step}, Run {entry.run}):")
    
    # Check Rmax_before_reduction
    if hasattr(entry, 'Rmax_before_reduction'):
        rmax = entry.Rmax_before_reduction
        if isinstance(rmax, np.ndarray) and rmax.size > 0:
            # Access nested struct
            rmax_obj = rmax.item() if hasattr(rmax, 'item') else rmax[0,0]
            if hasattr(rmax_obj, 'num_generators'):
                num_gen = rmax_obj.num_generators
                # Handle nested array
                if isinstance(num_gen, np.ndarray):
                    num_gen_val = num_gen.item() if num_gen.size == 1 else num_gen[0,0] if num_gen.size > 0 else None
                else:
                    num_gen_val = num_gen
                print(f"  Rmax before reduction: {num_gen_val} generators")
            else:
                print(f"  Rmax before reduction: struct without num_generators, fields: {dir(rmax_obj)}")
        else:
            print("  Rmax before reduction: NOT TRACKED (empty)")
    else:
        print("  Rmax before reduction: FIELD NOT FOUND")
    
    # Check Rlinti_before_Rmax
    if hasattr(entry, 'Rlinti_before_Rmax'):
        rlinti = entry.Rlinti_before_Rmax
        if isinstance(rlinti, np.ndarray) and rlinti.size > 0:
            rlinti_obj = rlinti.item() if hasattr(rlinti, 'item') else rlinti[0,0]
            if hasattr(rlinti_obj, 'num_generators'):
                num_gen = rlinti_obj.num_generators
                if isinstance(num_gen, np.ndarray):
                    num_gen_val = num_gen.item() if num_gen.size == 1 else num_gen[0,0] if num_gen.size > 0 else None
                else:
                    num_gen_val = num_gen
                print(f"  Rlinti before Rmax: {num_gen_val} generators")
            else:
                print(f"  Rlinti before Rmax: struct without num_generators")
        else:
            print("  Rlinti before Rmax: NOT TRACKED (empty)")
    else:
        print("  Rlinti before Rmax: FIELD NOT FOUND")
    
    # Check RallError_before_Rmax
    if hasattr(entry, 'RallError_before_Rmax'):
        rallerror = entry.RallError_before_Rmax
        if isinstance(rallerror, np.ndarray) and rallerror.size > 0:
            rallerror_obj = rallerror.item() if hasattr(rallerror, 'item') else rallerror[0,0]
            if hasattr(rallerror_obj, 'num_generators'):
                num_gen = rallerror_obj.num_generators
                if isinstance(num_gen, np.ndarray):
                    num_gen_val = num_gen.item() if num_gen.size == 1 else num_gen[0,0] if num_gen.size > 0 else None
                else:
                    num_gen_val = num_gen
                print(f"  RallError before Rmax: {num_gen_val} generators")
            else:
                print(f"  RallError before Rmax: struct without num_generators")
        else:
            print("  RallError before Rmax: NOT TRACKED (empty)")
    else:
        print("  RallError before Rmax: FIELD NOT FOUND")
    
    print()
    
    # Also check R_before_reduction for comparison
    if hasattr(entry, 'R_before_reduction'):
        r_before = entry.R_before_reduction
        if isinstance(r_before, np.ndarray) and r_before.size > 0:
            if isinstance(r_before, np.ndarray) and r_before.dtype.names:
                if 'num_generators' in r_before.dtype.names:
                    print(f"  R before reduction: {r_before['num_generators'].item()} generators")
                else:
                    print(f"  R before reduction: struct with fields {r_before.dtype.names}")
            else:
                print(f"  R before reduction: {type(r_before)}")
        else:
            print("  R before reduction: NOT TRACKED (empty)")
    else:
        print("  R before reduction: FIELD NOT FOUND")
else:
    print(f"ERROR: Entry {step_idx + 1} not found (only {len(upstream_log)} entries)")
