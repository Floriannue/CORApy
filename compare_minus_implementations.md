# Comparison: MATLAB vs Python Minus Implementation

## MATLAB Behavior (`contSet/minus.m`)

### Supported Operations:
1. **`Z - v`** (zonotope minus vector):
   - Implementation: `Z + (-v)` via `plus.m`
   - ✅ Supported

2. **`v - Z`** (vector minus zonotope):
   - Implementation: `-Z + v` via `uminus.m` + `plus.m`
   - ✅ Supported

3. **`Z1 - Z2`** (zonotope minus zonotope):
   - ❌ **NOT supported via `-` operator**
   - Must use `minkDiff(Z1, Z2)` explicitly
   - Error message: "Use minkDiff instead for set-set subtraction"

### MATLAB Code:
```matlab
function S = minus(S,p)
    if isnumeric(p)
        S = S + (-p);  % Z - v
    elseif isnumeric(S) && isa(p,'contSet')
        S = -p + S;    % v - Z
    else
        throw(CORAerror('CORA:notSupported',...
            'Use minkDiff instead for set-set subtraction'));
    end
end
```

## Python Current Implementation (`zonotope/minus.py`)

### Supported Operations:
1. **`Z - v`** (zonotope minus vector):
   - Implementation: `Z.plus(-v)` ✅ Matches MATLAB

2. **`v - Z`** (vector minus zonotope):
   - Implementation: `rminus(v, Z)` → `(-Z).plus(v)` ✅ Matches MATLAB

3. **`Z1 - Z2`** (zonotope minus zonotope):
   - Implementation: `Z1 + (-Z2)` ✅ Works but MATLAB doesn't allow this
   - This is an approximation of Minkowski difference

## Key Differences:

1. **`Z1 - Z2` Support**:
   - MATLAB: ❌ Not allowed via `-`, must use `minkDiff`
   - Python: ✅ Allowed, computes `Z1 + (-Z2)` (approximation)

2. **Right-sided subtraction**:
   - MATLAB: Uses `contSet/minus.m` dispatcher
   - Python: Uses `rminus` function (matches MATLAB logic)

## Recommendation:

The Python implementation should match MATLAB exactly:
- ✅ `Z - v` → `Z + (-v)` (correct)
- ✅ `v - Z` → `-Z + v` (correct)
- ⚠️ `Z1 - Z2` → Should raise error directing to `minkDiff` (currently allows it)

However, if tests expect `Z1 - Z2` to work, we may need to keep it but ensure it matches `minkDiff` behavior.

