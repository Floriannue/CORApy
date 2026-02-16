"""Compare Step 2's Rlintp and Rerror between Python and MATLAB."""
import pickle
import scipy.io
import numpy as np

# Load Python log
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

# Load MATLAB log
matlab_log = scipy.io.loadmat('upstream_matlab_log.mat', squeeze_me=True, struct_as_record=False)
matlab_upstream_log = matlab_log['upstreamLog']

# Find Step 2 entries (use Run 2)
python_step2 = None
for entry in python_log.get('upstreamLog', []):
    if entry.get('step') == 2 and entry.get('run') == 2:
        python_step2 = entry
        break

matlab_step2 = None
for entry in matlab_upstream_log:
    if hasattr(entry, 'step') and entry.step == 2:
        run = entry.run if hasattr(entry, 'run') else None
        if isinstance(run, np.ndarray):
            run = run.item() if run.size == 1 else run
        if run == 2:
            matlab_step2 = entry
            break

if python_step2 is None:
    print("ERROR: Could not find Python Step 2 entry")
    exit(1)

if matlab_step2 is None:
    print("ERROR: Could not find MATLAB Step 2 entry")
    exit(1)

print("=" * 80)
print("Step 2: Rlintp and Rerror Comparison")
print("=" * 80)
print()

# Compare Rlintp
print("--- Rlintp ---")
py_rlintp = python_step2.get('Rlintp_tracking')
matlab_rlintp = matlab_step2.Rlintp_tracking if hasattr(matlab_step2, 'Rlintp_tracking') else None

if py_rlintp is None:
    print("Python: Rlintp_tracking NOT FOUND")
else:
    py_num_gen = py_rlintp.get('num_generators', 0)
    print(f"Python: {py_num_gen} generators")

if matlab_rlintp is None or (isinstance(matlab_rlintp, np.ndarray) and matlab_rlintp.size == 0):
    print("MATLAB: Rlintp_tracking NOT FOUND or empty")
else:
    if isinstance(matlab_rlintp, np.ndarray):
        matlab_rlintp = matlab_rlintp.item()
    matlab_num_gen = matlab_rlintp.num_generators
    if isinstance(matlab_num_gen, np.ndarray):
        matlab_num_gen = matlab_num_gen.item() if matlab_num_gen.size == 1 else matlab_num_gen
    print(f"MATLAB: {matlab_num_gen} generators")
    
    if py_rlintp is not None:
        if py_num_gen == matlab_num_gen:
            print(f"  -> MATCH: Both have {py_num_gen} generators")
        else:
            print(f"  -> MISMATCH: Python {py_num_gen} vs MATLAB {matlab_num_gen} (diff: {matlab_num_gen - py_num_gen})")

print()

# Compare Rerror
print("--- Rerror ---")
py_rerror = python_step2.get('Rerror_tracking')
matlab_rerror = matlab_step2.Rerror_tracking if hasattr(matlab_step2, 'Rerror_tracking') else None

if py_rerror is None:
    print("Python: Rerror_tracking NOT FOUND")
else:
    py_num_gen = py_rerror.get('num_generators', 0)
    print(f"Python: {py_num_gen} generators")

if matlab_rerror is None or (isinstance(matlab_rerror, np.ndarray) and matlab_rerror.size == 0):
    print("MATLAB: Rerror_tracking NOT FOUND or empty")
else:
    if isinstance(matlab_rerror, np.ndarray):
        matlab_rerror = matlab_rerror.item()
    matlab_num_gen = matlab_rerror.num_generators
    if isinstance(matlab_num_gen, np.ndarray):
        matlab_num_gen = matlab_num_gen.item() if matlab_num_gen.size == 1 else matlab_num_gen
    print(f"MATLAB: {matlab_num_gen} generators")
    
    if py_rerror is not None:
        if py_num_gen == matlab_num_gen:
            print(f"  -> MATCH: Both have {py_num_gen} generators")
        else:
            print(f"  -> MISMATCH: Python {py_num_gen} vs MATLAB {matlab_num_gen} (diff: {matlab_num_gen - py_num_gen})")

print()

# Compare Rtp_final (before reduction in reach_adaptive)
print("--- Rtp (final, before reduction in reach_adaptive) ---")
py_rtp = python_step2.get('Rtp_final_tracking')
matlab_rtp = matlab_step2.Rtp_final_tracking if hasattr(matlab_step2, 'Rtp_final_tracking') else None

if py_rtp is None:
    print("Python: Rtp_final_tracking NOT FOUND")
else:
    py_num_gen = py_rtp.get('num_generators', 0)
    print(f"Python: {py_num_gen} generators")

if matlab_rtp is None or (isinstance(matlab_rtp, np.ndarray) and matlab_rtp.size == 0):
    print("MATLAB: Rtp_final_tracking NOT FOUND or empty")
else:
    if isinstance(matlab_rtp, np.ndarray):
        matlab_rtp = matlab_rtp.item()
    matlab_num_gen = matlab_rtp.num_generators
    if isinstance(matlab_num_gen, np.ndarray):
        matlab_num_gen = matlab_num_gen.item() if matlab_num_gen.size == 1 else matlab_num_gen
    print(f"MATLAB: {matlab_num_gen} generators")
    
    if py_rtp is not None:
        if py_num_gen == matlab_num_gen:
            print(f"  -> MATCH: Both have {py_num_gen} generators")
        else:
            print(f"  -> MISMATCH: Python {py_num_gen} vs MATLAB {matlab_num_gen} (diff: {matlab_num_gen - py_num_gen})")

print()
print("=" * 80)
print("Summary:")
print("=" * 80)
if py_rlintp and matlab_rlintp and py_rerror and matlab_rerror:
    py_rlintp_gen = py_rlintp.get('num_generators', 0)
    py_rerror_gen = py_rerror.get('num_generators', 0)
    py_expected_rtp = py_rlintp_gen + py_rerror_gen
    
    if isinstance(matlab_rlintp, np.ndarray):
        matlab_rlintp = matlab_rlintp.item()
    matlab_rlintp_gen = matlab_rlintp.num_generators
    if isinstance(matlab_rlintp_gen, np.ndarray):
        matlab_rlintp_gen = matlab_rlintp_gen.item() if matlab_rlintp_gen.size == 1 else matlab_rlintp_gen
    
    if isinstance(matlab_rerror, np.ndarray):
        matlab_rerror = matlab_rerror.item()
    matlab_rerror_gen = matlab_rerror.num_generators
    if isinstance(matlab_rerror_gen, np.ndarray):
        matlab_rerror_gen = matlab_rerror_gen.item() if matlab_rerror_gen.size == 1 else matlab_rerror_gen
    
    matlab_expected_rtp = matlab_rlintp_gen + matlab_rerror_gen
    
    print(f"Python: Rlintp ({py_rlintp_gen}) + Rerror ({py_rerror_gen}) = {py_expected_rtp} generators")
    print(f"MATLAB: Rlintp ({matlab_rlintp_gen}) + Rerror ({matlab_rerror_gen}) = {matlab_expected_rtp} generators")
    
    if py_rtp and matlab_rtp:
        py_rtp_gen = py_rtp.get('num_generators', 0)
        if isinstance(matlab_rtp, np.ndarray):
            matlab_rtp = matlab_rtp.item()
        matlab_rtp_gen = matlab_rtp.num_generators
        if isinstance(matlab_rtp_gen, np.ndarray):
            matlab_rtp_gen = matlab_rtp_gen.item() if matlab_rtp_gen.size == 1 else matlab_rtp_gen
        
        print(f"Python: Actual Rtp = {py_rtp_gen} generators")
        print(f"MATLAB: Actual Rtp = {matlab_rtp_gen} generators")
        
        if py_expected_rtp != py_rtp_gen:
            print(f"  -> Python: Expected {py_expected_rtp} but got {py_rtp_gen} (diff: {py_rtp_gen - py_expected_rtp})")
        if matlab_expected_rtp != matlab_rtp_gen:
            print(f"  -> MATLAB: Expected {matlab_expected_rtp} but got {matlab_rtp_gen} (diff: {matlab_rtp_gen - matlab_expected_rtp})")
