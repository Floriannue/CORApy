"""debug_matlab_quadmat - Debug MATLAB quadMat data extraction"""

import numpy as np
import scipy.io
import os

matlab_file = 'upstream_matlab_log.mat'
if os.path.exists(matlab_file):
    matlab_data = scipy.io.loadmat(matlab_file, squeeze_me=True)
    if 'upstreamLog' in matlab_data:
        matlab_upstream = matlab_data['upstreamLog']
        print(f"Loaded {len(matlab_upstream)} entries")
        
        # Find Step 3 entry
        for i in range(len(matlab_upstream)):
            e = matlab_upstream[i]
            if hasattr(e, 'dtype') and 'step' in e.dtype.names:
                step_val = e['step']
                step = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val
                if int(step) == 3:
                    print(f"\nStep 3 entry found at index {i}")
                    print(f"Entry type: {type(e)}")
                    print(f"Entry dtype: {e.dtype if hasattr(e, 'dtype') else 'N/A'}")
                    
                    # Check quadmat_tracking
                    if hasattr(e, 'dtype') and 'quadmat_tracking' in e.dtype.names:
                        qm = e['quadmat_tracking']
                        print(f"\nquadmat_tracking type: {type(qm)}")
                        print(f"quadmat_tracking shape: {qm.shape if hasattr(qm, 'shape') else 'N/A'}")
                        print(f"quadmat_tracking size: {qm.size if hasattr(qm, 'size') else 'N/A'}")
                        
                        if isinstance(qm, np.ndarray) and qm.size > 0:
                            print(f"quadmat_tracking[0] type: {type(qm[0])}")
                            if hasattr(qm[0], 'dtype'):
                                print(f"quadmat_tracking[0] dtype: {qm[0].dtype}")
                                print(f"quadmat_tracking[0] dtype.names: {qm[0].dtype.names if hasattr(qm[0].dtype, 'names') else 'N/A'}")
                                
                                if qm[0].dtype.names:
                                    for name in qm[0].dtype.names:
                                        val = qm[0][name]
                                        print(f"  {name}: type={type(val)}, shape={val.shape if hasattr(val, 'shape') else 'N/A'}, size={val.size if hasattr(val, 'size') else 'N/A'}")
                                        if name == 'dense_full' and hasattr(val, 'shape'):
                                            print(f"    dense_full value:\n{val}")
                    break
