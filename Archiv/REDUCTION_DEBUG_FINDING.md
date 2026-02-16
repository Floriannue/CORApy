# Reduction Debug Finding

## Root Cause Identified

The divergence in `Rend.tp` generator counts (Python: 2, MATLAB: 4) originates from the **adaptive reduction** in `initReach_adaptive`.

### Key Finding:
- **Rhom_tp (before reduction)**: Both Python and MATLAB have **5 generators** ✅ MATCH
- **Rend.tp (after reduction)**: Python has **2 generators**, MATLAB has **4 generators** ❌ MISMATCH
- **Reduction removed**: Python removed **3 generators**, MATLAB removed **1 generator**

### The Issue:
The `priv_reduceAdaptive` function is computing different `redIdx` values in Python vs MATLAB, causing different numbers of generators to be reduced.

### Next Steps:
1. Add debug tracking to `priv_reduceAdaptive` to capture:
   - `dHmax` value
   - `h` array values
   - `redIdx` computation
   - `gredIdx` result
2. Compare these values between Python and MATLAB to find where the computation diverges
3. Fix the Python implementation to match MATLAB

### Suspected Issues:
- `vecnorm` vs `np.linalg.norm` computation differences
- Index conversion between 0-based and 1-based
- Floating point precision differences affecting the `h <= dHmax` comparison
