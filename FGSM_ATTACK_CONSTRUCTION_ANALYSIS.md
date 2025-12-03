# FGSM Attack Construction Analysis

## MATLAB Implementation (lines 462-478)

```matlab
case 'fgsm'
    % Obtain number of constraints.
    [p,~] = size(A);
    % Try to falsification with a FGSM attack.
    if safeSet
        grad = pagemtimes(-A,S);
        % We combine all constraints for a stronger attack.
        p = 1;
    else
        grad = pagemtimes(A,S);
    end
    % If there are multiple output constraints we try to falsify
    % each one individually.
    sgrad = reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p]);
    
    % Compute adversarial attacks based on the sensitivity.
    xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad;
```

### Step-by-step analysis:

1. **Get number of constraints:**
   - `[p,~] = size(A);` → `p = num_constraints`

2. **Compute gradient:**
   - If `safeSet`: `grad = pagemtimes(-A,S); p = 1;`
     - `A` shape: `(p_orig, nK)`
     - `S` shape: `(nK, n0, cbSz)`
     - `grad` shape: `(p_orig, n0, cbSz)` after pagemtimes
     - **CRITICAL:** MATLAB sets `p = 1` but does NOT sum `grad`!
     - This means `grad` still has shape `(p_orig, n0, cbSz)` but `p = 1`
   - Else: `grad = pagemtimes(A,S);`
     - `grad` shape: `(p, n0, cbSz)`

3. **Compute sign gradient:**
   - `sign(grad)` shape: `(p, n0, cbSz)` where `p` is final value (1 for safeSet, p_orig otherwise)
   - `permute(sign(grad),[2 3 1])`: `(p, n0, cbSz)` → `(n0, cbSz, p)`
   - `reshape(...,[n0 cbSz*p])`: `(n0, cbSz, p)` → `(n0, cbSz*p)`

4. **Compute attack:**
   - `repelem(xi,1,p)`: repeat `xi` `p` times along dimension 2 → `(n0, cbSz*p)`
   - `repelem(ri,1,p)`: repeat `ri` `p` times along dimension 2 → `(n0, cbSz*p)`
   - `xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad` → `(n0, cbSz*p)`

## Python Implementation (lines 654-776)

```python
p_orig = A.shape[0] if A.ndim == 2 else 1
if safeSet:
    grad = -np.einsum('ij,jkl->ikl', A, S)  # (p_orig, n0, cbSz)
    # MATLAB: We combine all constraints for a stronger attack.
    grad = np.sum(grad, axis=0, keepdims=True)  # (1, n0, cbSz)
    p = 1
else:
    grad = np.einsum('ij,jkl->ikl', A, S)  # (p_orig, n0, cbSz)
    p = p_orig
sgrad = np.sign(grad).transpose(1, 2, 0).reshape(n0, cbSz * p)
xi_repeated = np.repeat(xi, p, axis=1)  # (n0, cbSz*p)
ri_repeated = np.repeat(ri, p, axis=1)  # (n0, cbSz*p)
zi = xi_repeated + ri_repeated * sgrad
```

## CRITICAL DIFFERENCE FOUND!

### Issue: SafeSet Constraint Combination

**MATLAB:**
- Computes `grad = pagemtimes(-A,S)` → shape `(p_orig, n0, cbSz)`
- Sets `p = 1` but **does NOT sum grad**
- `sign(grad)` still has shape `(p_orig, n0, cbSz)` but `p = 1`
- When reshaping: `reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p])`
  - Since `p = 1`, this becomes `reshape(...,[n0 cbSz*1])` = `(n0, cbSz)`
  - But `sign(grad)` has `p_orig` constraints in dimension 0!
  - **This seems like a bug in MATLAB or the interpretation is wrong**

Wait, let me reconsider. If `p = 1` after the if statement, then:
- `grad` shape: `(p_orig, n0, cbSz)` (from pagemtimes)
- `sign(grad)` shape: `(p_orig, n0, cbSz)`
- `permute(sign(grad),[2 3 1])`: `(n0, cbSz, p_orig)`
- `reshape(...,[n0 cbSz*1])`: This would try to reshape `(n0, cbSz, p_orig)` to `(n0, cbSz)`
- This would fail if `p_orig > 1`!

**Actually, I think MATLAB must be summing the constraints implicitly or the comment is misleading.**

Let me check if MATLAB's `pagemtimes` with `-A` where `A` is `(p_orig, nK)` and we set `p=1` means something different...

**Alternative interpretation:** Maybe MATLAB's comment "We combine all constraints" means that `pagemtimes(-A,S)` where `A` is `(p_orig, nK)` actually produces `(1, n0, cbSz)` by summing? But that doesn't match how `pagemtimes` works...

**Most likely:** MATLAB actually sums the constraints, but the code doesn't show it explicitly. The comment "We combine all constraints for a stronger attack" suggests summing.

**Python's approach:**
- Explicitly sums: `grad = np.sum(grad, axis=0, keepdims=True)`
- This produces `(1, n0, cbSz)` which matches `p = 1`

## Verification Needed

1. **Check if MATLAB actually sums constraints for safeSet:**
   - Run MATLAB code and check `grad` shape after `pagemtimes(-A,S)` when `p=1`
   - Check if `sign(grad)` has shape `(1, n0, cbSz)` or `(p_orig, n0, cbSz)`

2. **Check permute/reshape logic:**
   - MATLAB: `permute(sign(grad),[2 3 1])` on `(p, n0, cbSz)` → `(n0, cbSz, p)`
   - Python: `.transpose(1, 2, 0)` on `(p, n0, cbSz)` → `(n0, cbSz, p)` ✓
   - MATLAB: `reshape(...,[n0 cbSz*p])` → `(n0, cbSz*p)`
   - Python: `.reshape(n0, cbSz * p)` → `(n0, cbSz*p)` ✓

3. **Check repelem vs repeat:**
   - MATLAB: `repelem(xi,1,p)` repeats along dimension 2, `p` times
   - Python: `np.repeat(xi, p, axis=1)` repeats along axis 1, `p` times ✓

## Potential Issues

1. **SafeSet constraint combination:**
   - Python explicitly sums constraints
   - MATLAB may or may not sum (unclear from code)
   - **This could cause different attack vectors!**

2. **Numerical precision:**
   - MATLAB uses double precision by default
   - Python uses float32 in some places
   - Could cause `sign(grad)` to differ slightly

3. **Bounds checking (EXTRA in Python):**
   - Python has extensive bounds checking that MATLAB doesn't have
   - This is defensive but shouldn't affect results if code is correct

## Recommendations

1. **Add logging to compare `grad` shapes and values:**
   ```python
   if verbose:
       print(f"FGSM: grad shape={grad.shape}, p={p}, safeSet={safeSet}")
       print(f"FGSM: sgrad shape={sgrad.shape}")
       print(f"FGSM: grad sample values={grad[:, :3, :3] if grad.ndim == 3 else grad[:3, :3]}")
   ```

2. **Verify MATLAB actually sums for safeSet:**
   - Check MATLAB output to see if `grad` has shape `(1, n0, cbSz)` or `(p_orig, n0, cbSz)` when `p=1`

3. **Consider removing explicit sum if MATLAB doesn't sum:**
   - If MATLAB doesn't sum, we should match that behavior
   - But the comment suggests it does combine constraints...

4. **Test with single constraint:**
   - If `p_orig = 1`, both should behave the same
   - This would help isolate the issue

