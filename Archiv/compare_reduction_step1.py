"""
Compare reduction parameters between Python and MATLAB for Step 1 Run 1
"""
import pickle
import scipy.io
import numpy as np

# Load Python debug data
with open('initReach_debug.pkl', 'rb') as f:
    py_data = pickle.load(f)

# Find Step 1 Run 1
py_entry = None
for entry in py_data:
    if entry.get('step') == 1 and entry.get('run') == 1:
        py_entry = entry
        break

if py_entry is None:
    print("[ERROR] Step 1 Run 1 not found in Python data")
    exit(1)

# Load MATLAB debug data
mat_data = scipy.io.loadmat('initReach_debug.mat', squeeze_me=True, struct_as_record=False)
mat_entries = mat_data['initReach_debug']

# Find Step 1 Run 1 in MATLAB
mat_entry = None
for entry in mat_entries:
    if hasattr(entry, 'step') and entry.step.item() == 1 and hasattr(entry, 'run') and entry.run.item() == 1:
        mat_entry = entry
        break

if mat_entry is None:
    print("[ERROR] Step 1 Run 1 not found in MATLAB data")
    exit(1)

print("="*80)
print("Step 1 Run 1 - Rend.tp Reduction Comparison")
print("="*80)

# Compare basic parameters
print("\nBasic Parameters:")
print(f"  Python Rhom_tp generators: {py_entry.get('Rhom_tp_num_generators')}")
print(f"  Python Rend.tp generators: {py_entry.get('Rend_tp_num_generators')}")
if hasattr(mat_entry, 'Rhom_tp_num_generators'):
    print(f"  MATLAB Rhom_tp generators: {mat_entry.Rhom_tp_num_generators.item()}")
if hasattr(mat_entry, 'Rend_tp_num_generators'):
    print(f"  MATLAB Rend.tp generators: {mat_entry.Rend_tp_num_generators.item()}")

# Compare reduction parameters
print("\nReduction Parameters:")
py_diagpercent = py_entry.get('reduction_diagpercent')
py_dHmax = py_entry.get('reduction_dHmax')
py_nrG = py_entry.get('reduction_nrG')
py_last0Idx = py_entry.get('reduction_last0Idx')
py_redIdx = py_entry.get('reduction_redIdx')
py_redIdx_0based = py_entry.get('reduction_redIdx_0based')
py_dHerror = py_entry.get('reduction_dHerror')
py_gredIdx_len = py_entry.get('reduction_gredIdx_len')
py_h_computed = py_entry.get('reduction_h_computed')

print(f"\n  Python:")
print(f"    diagpercent: {py_diagpercent}")
print(f"    dHmax: {py_dHmax}")
print(f"    nrG: {py_nrG}")
print(f"    last0Idx: {py_last0Idx}")
print(f"    redIdx (1-based): {py_redIdx}")
print(f"    redIdx_0based: {py_redIdx_0based}")
print(f"    dHerror: {py_dHerror}")
print(f"    gredIdx_len: {py_gredIdx_len}")

# MATLAB reduction details
mat_diagpercent = None
mat_dHmax = None
mat_nrG = None
mat_last0Idx = None
mat_redIdx = None
mat_redIdx_0based = None
mat_dHerror = None
mat_gredIdx_len = None
mat_h_computed = None

if hasattr(mat_entry, 'reduction_diagpercent'):
    mat_diagpercent = mat_entry.reduction_diagpercent.item() if hasattr(mat_entry.reduction_diagpercent, 'item') else mat_entry.reduction_diagpercent
if hasattr(mat_entry, 'reduction_dHmax'):
    mat_dHmax = mat_entry.reduction_dHmax.item() if hasattr(mat_entry.reduction_dHmax, 'item') else mat_entry.reduction_dHmax
if hasattr(mat_entry, 'reduction_nrG'):
    mat_nrG = mat_entry.reduction_nrG.item() if hasattr(mat_entry.reduction_nrG, 'item') else mat_entry.reduction_nrG
if hasattr(mat_entry, 'reduction_last0Idx'):
    mat_last0Idx = mat_entry.reduction_last0Idx.item() if hasattr(mat_entry.reduction_last0Idx, 'item') else mat_entry.reduction_last0Idx
if hasattr(mat_entry, 'reduction_redIdx'):
    mat_redIdx = mat_entry.reduction_redIdx.item() if hasattr(mat_entry.reduction_redIdx, 'item') else mat_entry.reduction_redIdx
if hasattr(mat_entry, 'reduction_redIdx_0based'):
    mat_redIdx_0based = mat_entry.reduction_redIdx_0based.item() if hasattr(mat_entry.reduction_redIdx_0based, 'item') else mat_entry.reduction_redIdx_0based
if hasattr(mat_entry, 'reduction_dHerror'):
    mat_dHerror = mat_entry.reduction_dHerror.item() if hasattr(mat_entry.reduction_dHerror, 'item') else mat_entry.reduction_dHerror
if hasattr(mat_entry, 'reduction_gredIdx_len'):
    mat_gredIdx_len = mat_entry.reduction_gredIdx_len.item() if hasattr(mat_entry.reduction_gredIdx_len, 'item') else mat_entry.reduction_gredIdx_len
if hasattr(mat_entry, 'reduction_h_computed'):
    mat_h_computed = mat_entry.reduction_h_computed
    if hasattr(mat_h_computed, 'item'):
        mat_h_computed = mat_h_computed.item()
    if isinstance(mat_h_computed, np.ndarray):
        mat_h_computed = mat_h_computed.flatten()

print(f"\n  MATLAB:")
print(f"    diagpercent: {mat_diagpercent}")
print(f"    dHmax: {mat_dHmax}")
print(f"    nrG: {mat_nrG}")
print(f"    last0Idx: {mat_last0Idx}")
print(f"    redIdx (1-based): {mat_redIdx}")
print(f"    redIdx_0based: {mat_redIdx_0based}")
print(f"    dHerror: {mat_dHerror}")
print(f"    gredIdx_len: {mat_gredIdx_len}")

# Compare h arrays
print("\n" + "="*80)
print("h array comparison:")
if py_h_computed is not None and mat_h_computed is not None:
    py_h = np.asarray(py_h_computed).flatten()
    mat_h = np.asarray(mat_h_computed).flatten()
    
    print(f"  Python h shape: {py_h.shape}, values: {py_h}")
    print(f"  MATLAB h shape: {mat_h.shape}, values: {mat_h}")
    
    if py_h.shape == mat_h.shape:
        diff = np.abs(py_h - mat_h)
        max_diff = np.max(diff)
        print(f"  Max difference: {max_diff}")
        if max_diff > 1e-10:
            print(f"  [WARNING] h arrays differ!")
            print(f"    Differences: {diff}")
        else:
            print(f"  [OK] h arrays match")
    else:
        print(f"  [ERROR] h array shapes differ!")
    
    # Check how many values are <= dHmax
    if py_dHmax is not None:
        py_count = np.sum(py_h <= py_dHmax)
        print(f"  Python: {py_count} values <= dHmax ({py_dHmax})")
    if mat_dHmax is not None:
        mat_count = np.sum(mat_h <= mat_dHmax)
        print(f"  MATLAB: {mat_count} values <= dHmax ({mat_dHmax})")
else:
    print("  [ERROR] h arrays not available for comparison")

# Compare key values
print("\n" + "="*80)
print("Key Value Comparison:")
if py_dHmax is not None and mat_dHmax is not None:
    dHmax_diff = abs(py_dHmax - mat_dHmax)
    print(f"  dHmax difference: {dHmax_diff}")
    if dHmax_diff > 1e-10:
        print(f"  [WARNING] dHmax differs!")

if py_redIdx is not None and mat_redIdx is not None:
    redIdx_diff = abs(py_redIdx - mat_redIdx)
    print(f"  redIdx difference: {redIdx_diff}")
    if redIdx_diff != 0:
        print(f"  [ERROR] redIdx differs! This is the root cause!")
        print(f"    Python reduces {py_redIdx} generators")
        print(f"    MATLAB reduces {mat_redIdx} generators")
        print(f"    Difference: {py_redIdx - mat_redIdx} generators")

if py_gredIdx_len is not None and mat_gredIdx_len is not None:
    gredIdx_diff = abs(py_gredIdx_len - mat_gredIdx_len)
    print(f"  gredIdx_len difference: {gredIdx_diff}")
    if gredIdx_diff != 0:
        print(f"  [ERROR] gredIdx_len differs!")

print("\n" + "="*80)
