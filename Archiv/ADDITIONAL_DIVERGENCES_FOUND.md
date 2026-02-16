# Additional Divergences Found

## Summary

After investigating further, I found **additional sources of divergence** beyond the `reduce('adaptive')` issue:

## 1. **`pickedGenerators` vs `pickedGeneratorsFast` Mismatch**

### Issue
- **Python**: `priv_reduceGirard` uses `pickedGenerators`
- **MATLAB**: `priv_reduceGirard` uses `pickedGeneratorsFast`

### Impact
`pickedGeneratorsFast` has a more optimized algorithm:
- Handles `nReduced < nUnreduced` vs `nReduced >= nUnreduced` differently
- Uses `mink` for small reductions, `maxk` for large reductions
- This can cause **different generator selections** even for the same input

### Code Location
- **Python**: `cora_python/contSet/zonotope/private/priv_reduceMethods.py:31`
- **MATLAB**: `cora_matlab/contSet/@zonotope/private/priv_reduceGirard.m:34`

### Fix Required
1. Create `pickedGeneratorsFast` in Python (or check if it exists)
2. Update `priv_reduceGirard` to use `pickedGeneratorsFast` instead of `pickedGenerators`

## 2. **`pickedGenerators` Implementation Differences**

### Issue
Python's `pickedGenerators` doesn't match MATLAB's `pickedGeneratorsFast` logic:

**MATLAB `pickedGeneratorsFast`**:
```matlab
if nReduced < nUnreduced
    % pick generators with smallest h values to be reduced
    [~,indRed] = mink(h,nReduced);
else
    % pick generators with largest h values to be kept
    [~,indUnred] = maxk(fliplr(h),nUnreduced);
    indUnred = nrOfGens - indUnred + 1; % maintain ordering
end
```

**Python `pickedGenerators`**:
```python
# Always picks smallest h values to be reduced
indRed = np.argsort(h)[:nReduced]
# Doesn't handle nReduced >= nUnreduced case differently
```

### Impact
- When `nReduced >= nUnreduced`, Python and MATLAB will select **different generators**
- This causes different reduced sets → different `Z` → different `errorSec` → different `VerrorDyn`

## 3. **Potential Issues in `priv_reduceAdaptive` Implementation**

### Possible Issues
1. **Indexing in `penven` case**: The `redIdx` calculation might be incorrect
2. **Indexing in `girard` case**: The `redIdx` conversion from 0-based to 1-based might be wrong
3. **`gensred` slicing**: Need to verify `gensred[:, :redIdx]` matches MATLAB's `gensred(:,1:redIdx)`

### Status
- Implementation is complete but needs verification
- Test shows it runs but results need comparison with MATLAB

## 4. **Other Potential Issues**

### Placeholder Implementations Found
- `priv_reduceIdx`: Simplified implementation
- `priv_reduceMethE`, `priv_reduceMethF`: Simplified implementations
- `priv_reduceRedistribute`, `priv_reduceCluster`, `priv_reduceScott`, `priv_reduceValero`, `priv_reduceSadraddini`, `priv_reduceScale`: All simplified

**Impact**: These are less critical as they're not used in the adaptive algorithm, but could cause issues if used elsewhere.

## Recommended Fixes (Priority Order)

### Priority 1: **Fix `pickedGeneratorsFast`**
1. Create `pickedGeneratorsFast` in Python matching MATLAB exactly
2. Update `priv_reduceGirard` to use `pickedGeneratorsFast`
3. This affects all Girard reductions, which are used frequently

### Priority 2: **Verify `priv_reduceAdaptive` Indexing**
1. Compare reduction results with MATLAB for same inputs
2. Verify `gredIdx` matches MATLAB
3. Check if `dHerror` values match

### Priority 3: **Test Full Integration**
1. Run jetEngine test again after fixes
2. Compare upstream computations
3. Verify differences are reduced

## Next Steps

1. **Create `pickedGeneratorsFast`** matching MATLAB exactly
2. **Update `priv_reduceGirard`** to use it
3. **Test and verify** reduction results match MATLAB
4. **Re-run full comparison** to see impact
