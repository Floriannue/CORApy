"""Check dHmax computation to find why Python's is 45x larger"""
import pickle
import numpy as np

print("=" * 80)
print("CHECKING dHmax COMPUTATION")
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

if 'initReach_tracking' in py_entry:
    it = py_entry['initReach_tracking']
    rhom_tp_gens = np.asarray(it.get('Rhom_tp_generators'))
    redFactor = it.get('redFactor', 0.0005)
    
    print(f"\nPython Step 2 Run 2:")
    print(f"  redFactor: {redFactor}")
    
    # Compute dHmax step by step
    diagpercent = np.sqrt(redFactor)
    print(f"  diagpercent = sqrt({redFactor}) = {diagpercent:.12e}")
    
    Gabs = np.abs(rhom_tp_gens)
    print(f"  Gabs shape: {Gabs.shape}")
    print(f"  Gabs:\n{Gabs}")
    
    Gbox = np.sum(Gabs, axis=1, keepdims=True)
    print(f"  Gbox = sum(Gabs, axis=1): {Gbox.flatten()}")
    
    Gbox_sq = Gbox ** 2
    print(f"  Gbox^2: {Gbox_sq.flatten()}")
    
    Gbox_sq_sum = np.sum(Gbox_sq)
    print(f"  sum(Gbox^2): {Gbox_sq_sum:.12e}")
    
    sqrt_sum = np.sqrt(Gbox_sq_sum)
    print(f"  sqrt(sum(Gbox^2)): {sqrt_sum:.12e}")
    
    dHmax = (diagpercent * 2) * sqrt_sum
    print(f"  dHmax = (diagpercent * 2) * sqrt(sum(Gbox^2))")
    print(f"  dHmax = ({diagpercent:.12e} * 2) * {sqrt_sum:.12e}")
    print(f"  dHmax = {dHmax:.12e}")
    
    # Compare with MATLAB's value
    matlab_dHmax = 1.372892275513e-04
    print(f"\n  MATLAB dHmax: {matlab_dHmax:.12e}")
    print(f"  Ratio (Python/MATLAB): {dHmax / matlab_dHmax:.2f}x")
    
    # Check if Gbox is different
    print(f"\n  If MATLAB's dHmax is correct, then:")
    print(f"    sqrt(sum(Gbox^2))_matlab = {matlab_dHmax / (diagpercent * 2):.12e}")
    print(f"    sqrt(sum(Gbox^2))_python = {sqrt_sum:.12e}")
    print(f"    Ratio: {sqrt_sum / (matlab_dHmax / (diagpercent * 2)):.2f}x")
    
    # Check if diagpercent is different
    print(f"\n  If Gbox is the same, then:")
    print(f"    diagpercent_matlab = {matlab_dHmax / (2 * sqrt_sum):.12e}")
    print(f"    diagpercent_python = {diagpercent:.12e}")
    print(f"    Ratio: {diagpercent / (matlab_dHmax / (2 * sqrt_sum)):.2f}x")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("The dHmax difference explains why Python reduces more generators.")
print("With a larger dHmax, more generators satisfy h <= dHmax.")
print("Need to check if:")
print("  1. Gbox computation is different (different generator values)")
print("  2. diagpercent is different (different redFactor)")
print("  3. The MATLAB entry is from a different step/run")
