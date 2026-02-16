"""Trace Step 2 Run 2 reduction in detail"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("TRACING STEP 2 RUN 2 REDUCTION")
print("=" * 80)

# Load Python tracking
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        py_entry = entry
        break

if py_entry is None:
    print("[ERROR] Could not find Python Step 2 Run 2")
    exit(1)

print("[OK] Found Python Step 2 Run 2")

# Get Rhom_tp and Rend_tp
if 'initReach_tracking' in py_entry:
    it = py_entry['initReach_tracking']
    rhom_tp_num = it.get('Rhom_tp_num_generators')
    rend_tp_num = it.get('Rend_tp_num_generators')
    redFactor = it.get('redFactor')
    
    print(f"\nPython Step 2 Run 2:")
    print(f"  Rhom_tp generators: {rhom_tp_num}")
    print(f"  Rend_tp generators: {rend_tp_num}")
    print(f"  Reduced: {rhom_tp_num - rend_tp_num} generators")
    print(f"  redFactor: {redFactor}")
    
    # Get Rhom_tp generators to compute what dHmax should be
    rhom_tp_gens = it.get('Rhom_tp_generators')
    if rhom_tp_gens is not None:
        G = np.asarray(rhom_tp_gens)
        print(f"\n  Rhom_tp generator matrix shape: {G.shape}")
        
        # Compute dHmax
        diagpercent = np.sqrt(redFactor) if redFactor else 0.02236
        Gabs = np.abs(G)
        Gbox = np.sum(Gabs, axis=1, keepdims=True)
        dHmax = (diagpercent * 2) * np.sqrt(np.sum(Gbox ** 2))
        print(f"  Computed dHmax: {dHmax:.12e}")
        print(f"  diagpercent: {diagpercent:.12e}")

# Load MATLAB tracking
print("\n" + "=" * 80)
print("MATLAB STEP 2 RUN 2")
print("=" * 80)

matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

mat_entry = None
for entry in matlab_log:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 2 and entry.run == 2:
            mat_entry = entry
            break

if mat_entry is None:
    print("[WARNING] Could not find MATLAB Step 2 Run 2")
    print("Note: Step 2 Run 2 uses timeStepequalHorizon path in MATLAB")
else:
    print("[OK] Found MATLAB Step 2 Run 2")
    
    # Check if it has initReach_tracking
    if hasattr(mat_entry, 'initReach_tracking'):
        it = mat_entry.initReach_tracking
        if isinstance(it, np.ndarray) and it.size > 0:
            it = it[0]
        
        rhom_tp_num = getattr(it, 'Rhom_tp_num_generators', None)
        rend_tp_num = getattr(it, 'Rend_tp_num_generators', None)
        
        if rhom_tp_num is not None:
            print(f"\nMATLAB Step 2 Run 2:")
            print(f"  Rhom_tp generators: {rhom_tp_num}")
            print(f"  Rend_tp generators: {rend_tp_num}")
            if rhom_tp_num and rend_tp_num:
                print(f"  Reduced: {rhom_tp_num - rend_tp_num} generators")
        else:
            print("\n[WARNING] MATLAB initReach_tracking is empty")

# Check Rlintp
print("\n" + "=" * 80)
print("RLINTP COMPARISON")
print("=" * 80)

if 'Rlintp_tracking' in py_entry:
    py_rlintp = py_entry['Rlintp_tracking']
    print(f"Python Rlintp: {py_rlintp.get('num_generators')} generators")

if mat_entry and hasattr(mat_entry, 'Rlintp_tracking'):
    mat_rlintp = mat_entry.Rlintp_tracking
    if isinstance(mat_rlintp, np.ndarray) and mat_rlintp.size > 0:
        mat_rlintp = mat_rlintp[0]
    mat_num = getattr(mat_rlintp, 'num_generators', None)
    if mat_num is not None:
        print(f"MATLAB Rlintp: {mat_num} generators")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("The divergence occurs in the reduction step.")
print("Python reduces 5->2, MATLAB reduces 5->4.")
print("This means Python's redIdx is 3, MATLAB's redIdx is 1.")
print("The difference must be in:")
print("  1. dHmax computation (diagpercent or Gbox)")
print("  2. h array computation (gensdiag)")
print("  3. The comparison h <= dHmax")
