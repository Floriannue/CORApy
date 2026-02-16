# pickedGeneratorsFast Translation Verification

## Summary

Verified that `pickedGeneratorsFast` is correctly translated and used.

## Translation Comparison

### MATLAB Implementation
```matlab
if nReduced < nUnreduced
    [~,indRed] = mink(h,nReduced);
    idxRed = false(1,nrOfGens);
    idxRed(indRed) = true;
    idxUnred = ~idxRed;
else
    [~,indUnred] = maxk(fliplr(h),nUnreduced);
    indUnred = nrOfGens - indUnred + 1; % maintain ordering
    idxUnred = false(1,nrOfGens);
    idxUnred(indUnred) = true;
    idxRed = ~idxUnred;
    % Note: indRed is NOT set in this branch
end
```

### Python Implementation
```python
if nReduced < nUnreduced:
    indRed = np.argpartition(h, nReduced - 1)[:nReduced]
    indRed = indRed[np.argsort(h[indRed])]  # Sort by value
    idxRed = np.zeros(nrOfGens, dtype=bool)
    idxRed[indRed] = True
    idxUnred = ~idxRed
else:
    h_flipped = np.flip(h)
    indUnred_flipped = np.argpartition(h_flipped, len(h_flipped) - nUnreduced)[-nUnreduced:]
    indUnred_flipped = indUnred_flipped[np.argsort(h_flipped[indUnred_flipped])]
    indUnred = nrOfGens - 1 - indUnred_flipped  # Convert to 0-based
    indUnred = np.sort(indUnred)
    idxUnred = np.zeros(nrOfGens, dtype=bool)
    idxUnred[indUnred] = True
    idxRed = ~idxUnred
    # Note: indRed remains empty (matches MATLAB)
```

## Key Points

1. ✅ **Correct Logic**: Handles both `nReduced < nUnreduced` and `nReduced >= nUnreduced` cases
2. ✅ **Index Conversion**: Correctly converts from flipped indices to original indices
3. ✅ **indRed Behavior**: Matches MATLAB - only set when `nReduced < nUnreduced`, otherwise empty
4. ✅ **Boolean Arrays**: Correctly creates `idxRed` and `idxUnred` for splitting generators

## Usage Verification

### ✅ Correctly Used In:
- `priv_reduceGirard`: Uses `pickedGeneratorsFast` (matches MATLAB)

### ✅ Correctly Uses `pickedGenerators` In:
- `priv_reducePCA`: Uses `pickedGenerators` (matches MATLAB)
- Other methods that use `pickedGenerators` (matches MATLAB)

## Status

✅ **TRANSLATION COMPLETE AND VERIFIED**

The implementation correctly matches MATLAB's behavior:
- Same generator selection logic
- Same index handling
- Same return value structure
- Correctly used in `priv_reduceGirard`
