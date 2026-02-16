"""Check MATLAB log file structure"""
import scipy.io
import numpy as np

print("Checking MATLAB log file structure...")
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
print(f"Keys in .mat file: {list(matlab_data.keys())}")

for key in matlab_data.keys():
    if not key.startswith('__'):
        val = matlab_data[key]
        print(f"\n{key}:")
        print(f"  Type: {type(val)}")
        if isinstance(val, np.ndarray):
            print(f"  Shape: {val.shape}")
            print(f"  Dtype: {val.dtype}")
            if val.size > 0 and val.size < 10:
                print(f"  Value: {val}")
