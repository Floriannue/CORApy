"""Check if MATLAB log contains reduction details"""

import scipy.io
import numpy as np

# Load MATLAB log
data = scipy.io.loadmat('upstream_matlab_log.mat')
entries = data['upstreamLog'][0]

print(f'Total entries: {len(entries)}')

# Find Step 3 entries
step3 = [e for e in entries if e['step'][0,0] == 3]
print(f'Step 3 entries: {len(step3)}')

if step3:
    e = step3[-1]
    rred = e.get('Rred_after_reduction', None)
    print(f'\nRred_after_reduction present: {rred is not None}')
    
    if rred is not None and rred.size > 0:
        rd = rred[0,0]
        print(f'Has reduction_details: {"reduction_details" in rd.dtype.names if hasattr(rd, "dtype") else False}')
        if hasattr(rd, 'dtype') and 'num_generators' in rd.dtype.names:
            print(f'Num generators: {rd["num_generators"][0,0]}')
        
        # Check for reduction_details
        if hasattr(rd, 'dtype') and 'reduction_details' in rd.dtype.names:
            red_details = rd['reduction_details']
            if red_details.size > 0:
                details = red_details[0,0]
                print(f'\nReduction details found!')
                if hasattr(details, 'dtype'):
                    print(f'Fields: {details.dtype.names}')
                    if 'final_generators' in details.dtype.names:
                        print(f'Final generators: {details["final_generators"][0,0]}')
                    if 'redIdx' in details.dtype.names:
                        print(f'redIdx: {details["redIdx"][0,0]}')
        else:
            print('\nNo reduction_details found in Rred_after_reduction')
    else:
        print('Rred_after_reduction is empty')
else:
    print('No Step 3 entries found')
