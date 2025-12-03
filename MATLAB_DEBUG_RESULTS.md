# MATLAB Debug Results - FGSM Constraints

## Test Case: prop_2.vnnlib with ACASXU_run2a_1_2_batch_2000.onnx

### Constraint Reading (CORRECT ✓)

**A shape:** [4, 5]
```
A = [-1  1  0  0  0
     -1  0  1  0  0
     -1  0  0  1  0
     -1  0  0  0  1]
```

**b shape:** [4, 1]
```
b = [0
     0
     0
     0]
```

**safeSet:** 0 (unsafeSet)

**Interpretation:**
- Constraints: Y_1 <= Y_0, Y_2 <= Y_0, Y_3 <= Y_0, Y_4 <= Y_0
- This becomes: Y_1 - Y_0 <= 0, Y_2 - Y_0 <= 0, Y_3 - Y_0 <= 0, Y_4 - Y_0 <= 0
- **This matches Python exactly!** ✓

### Input Set

**xi (center):** [0.639929, 0.000000, 0.000000, 0.475000, -0.475000]
**ri (radius):** [0.039929, 0.500000, 0.500000, 0.025000, 0.025000]

### First Iteration FGSM Attack

**Mock S (for demonstration):** shape [5, 5, 1]

**Gradient computation:**
- `grad = pagemtimes(A, S)` → shape [4, 5, 1]
- `p = 4` (tries each constraint individually)
- `sgrad = sign(grad)` → shape [5, 4]
- Attack direction: `[1, 1, 1, 1, -1]` for first constraint
- **This moves in direction that INCREASES A*y**

**Attack candidates:**
- `xi_(:,1) = [0.679858, 0.500000, 0.500000, 0.500000, -0.500000]`

**Specification check:**
- `ld_yi(:,1) = [0.000976, 0.001128, 0.001372, 0.001365]` (all positive)
- `b = [0, 0, 0, 0]`
- `all(ld_yi <= b)` = False (no counterexample)
- **This matches Python behavior!**

### Verification Results

**MATLAB found COUNTEREXAMPLE after 13 iterations:**

```
Iteration 13: Queue=108, Verified=396, Total=504
Result: COUNTEREXAMPLE
```

**Counterexample:**
- `x_ = [0.600000, 0.015625, -0.218750, 0.450000, -0.450000]`
- `y_ = [-0.017436, -0.018235, -0.017746, -0.017623, -0.017841]`
- `A*y = [-0.000800, -0.000310, -0.000187, -0.000405]` (all negative)
- `all(A*y <= 0)` = True ✓

**Verification time:** 2.123 seconds

## Key Insights

1. **Constraint interpretation is CORRECT** - Python matches MATLAB exactly
2. **First iteration behavior matches** - Both find no counterexample initially
3. **MATLAB finds counterexample after splitting** - After 13 iterations
4. **The +grad direction for unsafeSet works** - But only after splitting creates the right input sets

## Critical Question

**Why does Python return VERIFIED when MATLAB finds COUNTEREXAMPLE?**

Possible reasons:
1. **Python terminates too early** - Queue becomes empty before finding counterexample
2. **Different splitting behavior** - Python explores different splits than MATLAB
3. **Numerical precision** - Small differences in splitting lead to different exploration
4. **Attack direction issue** - Python's attack doesn't find counterexamples even after splitting

## Python vs MATLAB Comparison

**Python Results:**
- Total iterations: **4**
- Queue becomes empty after 4 iterations
- All patches verified (33 verified patches)
- Returns **VERIFIED**

**MATLAB Results:**
- Total iterations: **13**
- Queue size at iteration 13: 108
- Verified: 396
- Returns **COUNTEREXAMPLE**

**Critical Finding:**
Python terminates **WAY TOO EARLY**! It verifies all patches after only 4 iterations, while MATLAB needs 13 iterations to find the counterexample.

## Root Cause Found! 🎯

**The Problem:** In iteration 4, `ld_ri` (radius) is **too small** (some are even 0!), causing patches to be verified incorrectly.

**Evidence from logs:**
- Iteration 4: `ld_yi = [0.01892354, 0.01912039, 0.00244045]` (positive)
- Iteration 4: `ld_ri = [0., 0.00018349, 0.10340225]` (very small, some are 0)
- Iteration 4: `ld_yi - ld_ri = [0.01892354, 0.0189369, -0.1009618]` (first two are positive)
- Check: `all(ld_yi - ld_ri <= 0)` = False → `unknown = False` (verified) ❌

**Why this is wrong:**
- If `ld_yi - ld_ri > 0`, the worst case (center - radius) still violates `A*y <= 0`
- The patch should be **unknown** (needs checking), not **verified**!
- But Python marks it as verified because `ld_ri` is too small

**The chain of issues:**
1. After splitting, input radius `ri` becomes very small
2. This causes input generators `Gxi` to be very small
3. This causes output generators `Gyi` to be very small
4. This causes `ld_ri = sum(abs(ld_Gyi)) + ld_Gyi_err` to be very small
5. When `ld_ri ≈ 0`, the check becomes `all(ld_yi <= 0)`
6. If `ld_yi > 0`, then `all(ld_yi <= 0)` = False → `unknown = False` (verified) ❌

**Root Cause Identified:**
- `Gyi` (output generators) becomes **all zeros** for some batches after splitting
- This causes `ld_ri = 0` (point zonotope)
- When `ld_ri = 0` and `ld_yi > 0` (center violates), the check `all(ld_yi - ld_ri <= 0)` becomes `all(ld_yi <= 0)` = False
- This incorrectly marks the patch as verified when it should remain unknown (needs counterexample search)

**Fix Applied:**
- Added special case: when `ld_ri ≈ 0` (point zonotope) and center violates (`ld_yi > b`), mark as unknown instead of verified
- This allows counterexample search to continue for violating points

**Remaining Questions:**
- Why does `Gyi` become all zeros? Is this expected behavior (network collapsing generators) or a bug?
- Does MATLAB handle this case the same way, or does it have different logic?
- Should we investigate why generators collapse to zero, or is the fix sufficient?

## Next Steps

1. **Add detailed logging for `ld_yi`, `ld_ri`, and `unknown` computation:**
   - Log values for first few iterations
   - Compare with MATLAB's values
   - Check if radius is too small

2. **Compare splitting behavior:**
   - Log which dimensions are split
   - Check if Python explores the same splits as MATLAB

3. **Check if Python finds the same counterexample:**
   - After similar splits, does Python find `x_ = [0.600000, 0.015625, -0.218750, 0.450000, -0.450000]`?
   - Or does it miss it because it terminates early?

4. **Test with negative grad:**
   - Try `fgsm_unsafe_direction = 'negative'` to see if it finds counterexamples earlier

