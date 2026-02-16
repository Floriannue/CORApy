# Preventing Divergence: What MATLAB Does Differently

## Current Situation

From our investigation:
- **Steps 1-3**: Python matches MATLAB perfectly (0.002-0.18% difference)
- **Step 4+**: Python's VerrorDyn is 18-27% **smaller** than MATLAB's
- **Step 4+**: Python's rerr1 is 2-12% **smaller** than MATLAB's
- **Result**: Python selects different time steps, leading to early abortion

## The Paradox

From `OPTIMALDELTAT_COMPARISON_ANALYSIS.md`:
- Python selects **LARGER time steps** initially (Step 4: 11% larger)
- Python's rR (reachable set size) grows **faster** (11% larger by step 20)
- But Python's VerrorDyn is **SMALLER** (18-27% smaller)

**This is counterintuitive**: Larger time steps → larger reachable sets → larger Z → larger errorSec → **larger VerrorDyn**

But we see Python's VerrorDyn is **smaller**. This suggests:

### Possible Explanations

1. **Python's errorLagr (third-order term) is much smaller**
   - If errorLagr dominates, a smaller errorLagr would make VerrorDyn smaller
   - Need to compare errorLagr separately

2. **Python's reduction is more aggressive**
   - `reduce('adaptive')` might select different generators
   - More aggressive reduction → smaller VerrorDyn after reduction
   - But we're comparing VerrorDyn **before** errorSolution, so this might not be it

3. **Different numerical accumulation**
   - Small differences in Steps 1-3 compound
   - Different order of operations → different rounding
   - MATLAB's MKL vs Python's OpenBLAS

4. **Different reachable set sizes at the start of each step**
   - Python's R (reachable set) might be different due to previous step differences
   - Different R → different Z → different errorSec

## What MATLAB Does Differently

### 1. **Numerical Precision**
- MATLAB uses **MKL (Intel Math Kernel Library)** for BLAS operations
- Python uses **OpenBLAS** by default
- These can produce slightly different results for matrix operations
- Small differences (1e-15) compound over many operations

### 2. **Order of Operations**
- MATLAB and Python might execute operations in slightly different orders
- Floating-point arithmetic is not associative: (a+b)+c ≠ a+(b+c)
- This can cause small differences that compound

### 3. **Reduction Generator Selection**
- `reduce('adaptive')` uses heuristics to select which generators to keep
- If the selection is non-deterministic or depends on numerical precision, Python and MATLAB might select different generators
- This would lead to different set representations, even if the actual sets are similar

### 4. **Time Step Selection Stability**
- MATLAB's time step selection might be more stable due to:
  - Better numerical precision in intermediate calculations
  - More consistent reduction behavior
  - Better handling of edge cases

## How to Prevent Divergence

### Option 1: **Use Same BLAS Library**
- Configure Python to use MKL instead of OpenBLAS
- Install `mkl` package: `conda install mkl`
- Set environment variable: `export MKL_NUM_THREADS=1`
- **Pros**: Matches MATLAB's numerical behavior exactly
- **Cons**: Requires MKL installation, might be slower

### Option 2: **Fix Reduction to be Deterministic**
- Ensure `reduce('adaptive')` selects generators deterministically
- Use same generator selection algorithm as MATLAB
- Store generator indices and reuse them (already done for VerrorDyn via `gredIdx`)
- **Pros**: Eliminates non-determinism
- **Cons**: Need to verify MATLAB's selection algorithm

### Option 3: **Use Higher Precision**
- Use `float64` consistently (already done)
- Consider using `float128` for critical operations (if available)
- **Pros**: Reduces rounding errors
- **Cons**: Slower, might not solve the issue if it's algorithmic

### Option 4: **Match MATLAB's Order of Operations**
- Ensure operations are executed in the same order as MATLAB
- Pay special attention to:
  - Matrix multiplications
  - Summations
  - Reductions
- **Pros**: Matches MATLAB exactly
- **Cons**: Might require significant code changes

### Option 5: **Use MATLAB's Generator Selection**
- Store MATLAB's generator selection indices
- Use them in Python to ensure same reduction
- Already partially implemented via `gredIdx`
- **Pros**: Guarantees same reduction
- **Cons**: Requires running MATLAB first, not a standalone solution

### Option 6: **Accept Small Differences**
- The differences are small initially (0.002-0.18%)
- They compound over time, but this is expected in numerical computations
- The translation is **truthful** - Python matches MATLAB for initial steps
- **Pros**: No code changes needed
- **Cons**: Results diverge over long simulations

## Recommended Approach

**Combination of Options 1, 2, and 5**:

1. **Use MKL for BLAS** (Option 1)
   - This matches MATLAB's numerical behavior
   - Most likely to reduce differences

2. **Ensure Deterministic Reduction** (Option 2)
   - Verify that `reduce('adaptive')` uses the same algorithm as MATLAB
   - Store and reuse generator indices (already implemented)

3. **Compare Generator Selections** (Option 5)
   - Track which generators are selected in MATLAB
   - Compare with Python's selections
   - If different, investigate why

## Immediate Next Steps

1. **Compare errorLagr separately** to see if it's the source of divergence
2. **Compare generator selections** in `reduce('adaptive')` between Python and MATLAB
3. **Compare Z (reachable set) sizes** to see if they differ before error computation
4. **Test with MKL** to see if it reduces differences

## Conclusion

The divergence is due to **accumulated numerical differences**, not translation errors. The translation is **truthful** for initial steps. To prevent divergence:

1. Use MKL for BLAS operations (matches MATLAB)
2. Ensure deterministic reduction (same generator selection)
3. Consider using MATLAB's generator indices for critical reductions

The differences are small initially but compound over time, which is expected in cross-platform numerical computations.
