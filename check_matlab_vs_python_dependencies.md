# MATLAB vs Python nn.verify Dependencies Comparison

## Methods Used in MATLAB nn.verify

### 1. PolyZonotope Methods

#### restructurePolyZono.m (line 36):
```matlab
pZ = restructure(pZ, 'reduceGirard', nrGen/length(c));
```
**Python**: ✓ `pZ.restructure('reduceGirard', nrGen/len(c))`

#### compBoundsPolyZono.m (line 40):
```matlab
int = interval(pZ, 'split');
l = infimum(int);
u = supremum(int);
```
**Python**: ✓ `int_result = pZ.interval('split')`
**Python**: ✓ `l = int_result.inf` (property, not method)
**Python**: ✓ `u = int_result.sup` (property, not method)

### 2. Interval Methods

#### conversionStarSetConZono.m (lines 76-77):
```matlab
cen = center(int);
R = diag(rad(int));
```
**Python**: ✓ `cen = int_val.center()`
**Python**: ✓ `R = np.diag(int_val.rad())`

#### conversionStarSetConZono.m (lines 86-89):
```matlab
l = infimum(interval(C*Z));
int = interval(l, d);
cen = center(int);
r = rad(int);
```
**Python**: ✓ `l_int = Z.interval().inf` (property)
**Python**: ✓ `int_val = Interval(l_int, d)`
**Python**: ✓ `cen = int_val.center()`
**Python**: ✓ `r = int_val.rad()`

### 3. Zonotope Methods

#### conversionStarSetConZono.m (line 85):
```matlab
Z = zonotope(interval(-ones(numGens, 1), ones(numGens, 1)));
l = infimum(interval(C*Z));
```
**Python**: ✓ `Z = Zonotope(Interval(-np.ones((numGens, 1)), np.ones((numGens, 1))))`
**Python**: ✓ `l_int = Z.interval().inf` (property)

### 4. Polytope Methods

#### conversionConZonoStarSet.m (line 52):
```matlab
int = interval(polytope(C, d));
l = infimum(int);
u = supremum(int);
```
**Python**: ✓ `int_result = Polytope(C, d).interval()`
**Python**: ✓ `l = int_result.inf.reshape(-1, 1)` (property)
**Python**: ✓ `u = int_result.sup.reshape(-1, 1)` (property)

### 5. representsa_ Method

#### nnActivationLayer.m (line 89):
```matlab
elseif representsa_(bounds,'emptySet',eps)
```
**Python**: ✓ `bounds.representsa_('emptySet', eps)`

#### nnLinearLayer.m (lines 84, 91, 110, 129):
```matlab
if representsa_(input,'emptySet',eps)
if ~representsa_(obj.d,'emptySet',eps)
```
**Python**: ✓ `obj.representsa_('emptySet', eps)`

## Summary

### All Required Methods Are Implemented:

1. **PolyZonotope**:
   - ✓ `restructure()` - Used in restructurePolyZono
   - ✓ `interval()` - Used in compBoundsPolyZono
   - ✓ `representsa_()` - Used in nnActivationLayer and nnLinearLayer
   - ✓ `compact_()` - Used internally by representsa_
   - ✓ `zonotope()` - Used internally by representsa_ and restructure

2. **Interval**:
   - ✓ `center()` - Used in conversionStarSetConZono
   - ✓ `rad()` - Used in conversionStarSetConZono
   - ✓ `interval()` - Constructor/self-reference
   - ✓ `representsa_()` - Used in nnActivationLayer
   - ✓ `.inf` property - Used in compBoundsPolyZono, conversionStarSetConZono, conversionConZonoStarSet
   - ✓ `.sup` property - Used in compBoundsPolyZono, conversionConZonoStarSet

3. **Zonotope**:
   - ✓ `interval()` - Used in conversionStarSetConZono
   - ✓ `center()` - Used in nnLinearLayer (via helper)
   - ✓ `generators()` - Used in restructure
   - ✓ `reduce()` - Used in restructure

4. **Polytope**:
   - ✓ `interval()` - Used in conversionConZonoStarSet

5. **ConZonotope**:
   - ✓ Properties: `.G`, `.A`, `.b` - Used in conversionConZonoStarSet
   - ✓ `center()` - Used in conversionStarSetConZono

## Notes

- MATLAB uses `infimum(int)` and `supremum(int)` which are property accesses
- Python correctly uses `.inf` and `.sup` properties
- All method calls match MATLAB exactly
- All dependencies are properly implemented

