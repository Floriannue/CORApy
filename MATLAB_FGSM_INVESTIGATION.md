# MATLAB FGSM Attack Investigation

## Key Questions

1. **Does MATLAB try both attack directions (+ and -)?**
   - **Answer: NO** - MATLAB only uses `xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad`
   - Always uses `+` sign, never tries `-`

2. **What happens for safeSet when p=1 but grad has multiple constraints?**
   - MATLAB code: `grad = pagemtimes(-A,S); p = 1;`
   - Then: `sgrad = reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p]);`
   - If `grad` has shape `(p_orig, n0, cbSz)` and `p=1`:
     - `sign(grad)` shape: `(p_orig, n0, cbSz)`
     - `permute(...,[2 3 1])` → `(n0, cbSz, p_orig)`
     - `reshape(...,[n0 cbSz*1])` → tries to reshape `(n0, cbSz, p_orig)` to `(n0, cbSz)`
     - **This would FAIL if p_orig > 1!**
   - **Conclusion:** MATLAB must be summing the constraints implicitly, OR the reshape is taking only the first constraint

3. **Why does MATLAB use `+grad` for unsafeSet when we want to decrease A*y?**
   - For `unsafeSet`: we want `A*y <= b` (decrease A*y)
   - MATLAB uses: `grad = pagemtimes(A,S)` (increases A*y)
   - Attack: `xi_ = xi + ri * sign(grad)` (moves in direction that increases A*y)
   - **This seems backwards!** We should use `-grad` to decrease A*y
   - **Possible explanations:**
     a) MATLAB code has a bug
     b) There's a different interpretation (maybe trying to find worst case?)
     c) The logic is correct for some reason we're missing

## MATLAB Code Analysis

### For safeSet:
```matlab
grad = pagemtimes(-A,S);  % Negative A
p = 1;  % Combine all constraints
sgrad = reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p]);
xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad;
```

### For unsafeSet:
```matlab
grad = pagemtimes(A,S);  % Positive A
% p stays as p_orig (tries each constraint individually)
sgrad = reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p]);
xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad;
```

## Critical Observation

**For unsafeSet:**
- We want: `A*y <= b` (find points where this is true)
- Current: `A*y > b` (we're in the safe region)
- To get into unsafe region, we need to **decrease** `A*y`
- But MATLAB uses `grad = A*S` which **increases** `A*y`
- Attack direction: `xi + ri * sign(grad)` moves **away** from unsafe region!

**This suggests MATLAB might have a bug, OR:**
- The attack is trying to find the "worst case" by maximizing A*y
- If even the worst case doesn't violate (A*y <= b is still false), then we know it's safe
- But this doesn't help with falsification - we want to find counterexamples!

## Hypothesis

Maybe MATLAB's logic is:
1. Try to maximize `A*y` (push as far as possible)
2. If `A*y <= b` is still false, try the opposite direction (minimize `A*y`)
3. But the code doesn't show this - it only shows one direction

**OR** maybe the interpretation is different:
- For `unsafeSet`, `A*y <= b` means the output is in the "unsafe" region
- The attack tries to push INTO this region
- But using `+grad` pushes AWAY from it...

## MATLAB Flow Analysis

Looking at lines 515-522:
```matlab
[~,critVal,falsified,x_,y_] = aux_checkPoints(nn,options,idxLayer,A,b,safeSet,xi_);

if any(falsified)
    % Found a counterexample.
    res.str = 'COUNTEREXAMPLE';
    break;
end
```

**Key observations:**
1. MATLAB computes FGSM attack: `xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad`
2. For `unsafeSet` with multiple constraints, `p = p_orig`, so it creates `p_orig` attack candidates
3. Each candidate tries to falsify one constraint individually
4. MATLAB checks if ANY candidate violates the spec
5. If yes, returns COUNTEREXAMPLE immediately

**The critical question:** Why does MATLAB use `+grad` for `unsafeSet` when we want to decrease `A*y`?

## Hypothesis: Maybe MATLAB's Logic is Different

**Alternative interpretation:**
- For `unsafeSet`, `A*y <= b` defines the "unsafe" region
- The FGSM attack tries to find the "worst case" by maximizing `A*y`
- If even the worst case (maximum `A*y`) doesn't satisfy `A*y <= b`, then we know it's safe
- But wait... this doesn't help with falsification! We want to find counterexamples, not prove safety.

**OR maybe:**
- The attack direction is actually correct for some reason we're missing
- Perhaps the semantics of `A` and `b` are different than we think
- Or maybe there's a sign error in our interpretation

## Constraint Interpretation Verification

**For prop_2:**
- VNNLIB: `Y_1 <= Y_0, Y_2 <= Y_0, Y_3 <= Y_0, Y_4 <= Y_0`
- This becomes: `Y_1 - Y_0 <= 0, Y_2 - Y_0 <= 0, Y_3 - Y_0 <= 0, Y_4 - Y_0 <= 0`
- Expected A: `[[-1, 1, 0, 0, 0], [-1, 0, 1, 0, 0], [-1, 0, 0, 1, 0], [-1, 0, 0, 0, 1]]`
- Expected b: `[0, 0, 0, 0]`
- MATLAB test confirms: `specs.set.A = [-ones(4,1) eye(4)]`, `specs.set.b = zeros(4,1)`
- **Our interpretation is CORRECT!**

**For unsafeSet:**
- We want: `all(A*y <= b)` to be True (find counterexamples)
- Currently: `A*y > b` (we're in safe region)
- To get into unsafe region: need to **decrease** `A*y` until `A*y <= b`
- But MATLAB uses: `grad = A*S` which **increases** `A*y`
- Attack: `xi_ = xi + ri * sign(grad)` moves **away** from unsafe region!

**This is still confusing!** The constraint interpretation is correct, but the attack direction seems backwards.

## Possible Explanations

1. **MATLAB might be exploring the "worst case" first:**
   - Try to maximize `A*y` (push as far as possible)
   - If even the worst case doesn't satisfy `A*y <= b`, then we know it's safe
   - But this doesn't help with falsification - we want to find counterexamples!

2. **Maybe the attack works differently:**
   - The attack might be trying to find points on the boundary
   - By pushing in one direction, we might cross the boundary
   - But this only works if we start on the "other side" of the boundary

3. **Maybe there's a sign error in MATLAB:**
   - The code might have a bug
   - Or the interpretation of `A` and `b` might be different than we think

4. **Maybe the attack is correct but we're missing something:**
   - Perhaps the splitting/refinement eventually creates input sets where the attack direction works
   - Or maybe the attack is trying to find the "most promising" direction for future splits

## MATLAB Debug Results (from debug_matlab_fgsm_constraints.m)

**Key Findings:**

1. **Constraint reading is CORRECT:**
   - A = [[-1, 1, 0, 0, 0], [-1, 0, 1, 0, 0], [-1, 0, 0, 1, 0], [-1, 0, 0, 0, 1]]
   - b = [0, 0, 0, 0]
   - This matches Python exactly ✓

2. **First iteration: NO counterexample found (matches Python):**
   - Mock S shows: `ld_yi(:,1) = [0.000976 0.001128 0.001372 0.001365]`
   - All positive, so `all(ld_yi <= b)` = False
   - No counterexamples found in first iteration
   - **This matches Python behavior!**

3. **After splitting: MATLAB finds COUNTEREXAMPLE:**
   - After 13 iterations of splitting
   - Counterexample: `x_ = [0.600000 0.015625 -0.218750 0.450000 -0.450000]`
   - `A*y = [-0.000800 -0.000310 -0.000187 -0.000405]` (all negative)
   - `all(A*y <= 0)` = True ✓

4. **Critical insight:**
   - MATLAB uses `+grad` for unsafeSet (increases A*y)
   - First iteration doesn't find counterexample
   - **But after splitting, the attack direction works!**
   - This suggests: splitting creates input sets where the +grad direction helps find counterexamples
   - OR: the attack explores the boundary in a way that eventually finds violations

## Hypothesis

**The attack direction `+grad` for unsafeSet works because:**
1. After splitting, some input sets are closer to the boundary
2. Moving in the `+grad` direction (increasing A*y) from these split sets can cross the boundary
3. Even though we want A*y <= b, moving away from the boundary first, then splitting, creates sets where the attack finds counterexamples

**OR:**
- The attack is exploring the "worst case" by maximizing A*y
- If even the worst case doesn't violate, we know it's safe
- But if splitting creates sets where the worst case DOES violate, we found a counterexample

## Next Steps

1. **Check Python's iteration count:**
   - Does Python run the same number of iterations?
   - Does it terminate early?
   - Add logging to show iteration count and queue size

2. **Check if Python finds the same counterexample:**
   - After the same number of splits, does Python find it?
   - Or does it terminate as VERIFIED before finding it?

3. **Test with -grad for unsafeSet:**
   - Use option `fgsm_unsafe_direction = 'negative'`
   - See if this finds counterexamples earlier
   - Compare iteration count with MATLAB

4. **Verify the reshape logic for safeSet:**
   - If `p_orig > 1` and `p = 1`, how does MATLAB handle the reshape?
   - Does it sum, take first, or something else?
   - This is critical for understanding the safeSet attack

