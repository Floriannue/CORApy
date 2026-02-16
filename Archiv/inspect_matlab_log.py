"""Inspect MATLAB log structure"""

import scipy.io
import numpy as np

# Load MATLAB log (use struct_as_record=True to get structured arrays)
data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=True)

print("Keys in data:", list(data.keys()))
print()

if 'upstreamLog' in data:
    entries = data['upstreamLog']
    print(f"upstreamLog type: {type(entries)}")
    print(f"upstreamLog shape: {entries.shape if hasattr(entries, 'shape') else 'N/A'}")
    print()
    
    # Access entries as structured array
    if entries.size > 0:
        print(f"Number of entries: {entries.size}")
        print(f"Entry dtype: {entries.dtype}")
        print(f"Entry dtype names: {entries.dtype.names}")
        print()
        
        # Find Step 3 entries
        step3_indices = []
        for i in range(min(entries.size, 10)):  # Check first 10
            entry = entries[i, 0]
            if 'step' in entries.dtype.names:
                step_val = entry['step'][0, 0] if entry['step'].size > 0 else None
                if step_val == 3:
                    step3_indices.append(i)
                print(f"Entry {i}: step={step_val}")
        
        print(f"\nStep 3 indices: {step3_indices}")
        
        if step3_indices:
            i = step3_indices[-1]
            entry = entries[i, 0]
            print(f"\n=== Entry {i} (Step 3) ===")
            
            if 'Rred_after_reduction' in entries.dtype.names:
                rred = entry['Rred_after_reduction']
                print(f"Rred_after_reduction type: {type(rred)}")
                print(f"Rred_after_reduction shape: {rred.shape if hasattr(rred, 'shape') else 'N/A'}")
                
                if rred.size > 0:
                    rred_struct = rred[0, 0]
                    print(f"Rred struct dtype: {rred_struct.dtype if hasattr(rred_struct, 'dtype') else 'N/A'}")
                    if hasattr(rred_struct, 'dtype') and rred_struct.dtype.names:
                        print(f"Rred fields: {rred_struct.dtype.names}")
                        
                        if 'reduction_details' in rred_struct.dtype.names:
                            rd = rred_struct['reduction_details']
                            print(f"\nreduction_details found!")
                            print(f"  Type: {type(rd)}")
                            print(f"  Shape: {rd.shape if hasattr(rd, 'shape') else 'N/A'}")
                            if rd.size > 0:
                                rd_struct = rd[0, 0]
                                if hasattr(rd_struct, 'dtype') and rd_struct.dtype.names:
                                    print(f"  Fields: {rd_struct.dtype.names}")
                                    for field in rd_struct.dtype.names:
                                        val = rd_struct[field]
                                        if val.size > 0:
                                            print(f"    {field}: {val[0, 0] if val.size == 1 else val.shape}")
                        else:
                            print("\nNo reduction_details field found")
                    else:
                        print("Rred is not a structured array")
                else:
                    print("Rred_after_reduction is empty")
