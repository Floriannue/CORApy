"""Compare Rmax, Rlinti, and RallError between Python and MATLAB"""

import pickle
import scipy.io
import numpy as np

# Load logs
print("Loading logs...")
with open('upstream_python_log.pkl', 'rb') as f:
    py_data = pickle.load(f)
ml_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=True)

py_entries = py_data.get('upstreamLog', [])
ml_entries = ml_data['upstreamLog']

# Find Step 3
step = 3
py_step3 = [e for e in py_entries if e.get('step') == step]
ml_step3_indices = [i for i in range(ml_entries.size) if ml_entries[i, 0]['step'][0, 0] == step]

if py_step3 and ml_step3_indices:
    py = py_step3[-1]
    ml_idx = ml_step3_indices[-1]
    ml = ml_entries[ml_idx, 0]
    
    print(f"\n{'='*80}")
    print(f"COMPARING RMAX COMPONENTS - Step {step}")
    print(f"{'='*80}\n")
    
    # Compare Rmax (this is R before reduction in priv_abstractionError_adaptive)
    print("RMAX (R before reduction in priv_abstractionError_adaptive):")
    py_rmax = py.get('Rmax_before_reduction')
    ml_rmax = ml['Rmax_before_reduction'] if 'Rmax_before_reduction' in ml.dtype.names else None
    
    if py_rmax and ml_rmax is not None and ml_rmax.size > 0:
        ml_rmax_struct = ml_rmax[0, 0]
        py_num = py_rmax.get('num_generators')
        ml_num = ml_rmax_struct['num_generators'][0, 0] if 'num_generators' in ml_rmax_struct.dtype.names else None
        print(f"  Python: {py_num} generators")
        print(f"  MATLAB: {ml_num} generators")
        print(f"  Difference: {abs(py_num - ml_num) if py_num and ml_num else 'N/A'} generators\n")
    else:
        print("  Rmax not tracked in one or both logs\n")
    
    # Compare Rlinti
    print("RLINTI (linearized reachable set):")
    py_rlinti = py.get('Rlinti_before_Rmax')
    ml_rlinti = ml['Rlinti_before_Rmax'] if 'Rlinti_before_Rmax' in ml.dtype.names else None
    
    if py_rlinti and ml_rlinti is not None and ml_rlinti.size > 0:
        ml_rlinti_struct = ml_rlinti[0, 0]
        py_num = py_rlinti.get('num_generators')
        ml_num = ml_rlinti_struct['num_generators'][0, 0] if 'num_generators' in ml_rlinti_struct.dtype.names else None
        print(f"  Python: {py_num} generators")
        print(f"  MATLAB: {ml_num} generators")
        print(f"  Difference: {abs(py_num - ml_num) if py_num and ml_num else 'N/A'} generators\n")
    else:
        print("  Rlinti not tracked in one or both logs\n")
    
    # Compare RallError
    print("RALLERROR (abstraction error solution):")
    py_rallerror = py.get('RallError_before_Rmax')
    ml_rallerror = ml['RallError_before_Rmax'] if 'RallError_before_Rmax' in ml.dtype.names else None
    
    if py_rallerror and ml_rallerror is not None and ml_rallerror.size > 0:
        ml_rallerror_struct = ml_rallerror[0, 0]
        py_num = py_rallerror.get('num_generators')
        ml_num = ml_rallerror_struct['num_generators'][0, 0] if 'num_generators' in ml_rallerror_struct.dtype.names else None
        print(f"  Python: {py_num} generators")
        print(f"  MATLAB: {ml_num} generators")
        print(f"  Difference: {abs(py_num - ml_num) if py_num and ml_num else 'N/A'} generators\n")
    else:
        print("  RallError not tracked in one or both logs\n")
    
    print(f"{'='*80}")
    print("ANALYSIS:")
    print("  Rmax = Rlinti + RallError")
    print("  If Rmax has different generator counts, check:")
    print("    1. Rlinti generator count")
    print("    2. RallError generator count")
    print("    3. Addition operation (cartProd or Minkowski sum)")
