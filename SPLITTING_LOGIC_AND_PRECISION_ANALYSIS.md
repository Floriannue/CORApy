# Splitting Logic and Double Precision Analysis

## Splitting Logic: `ri_` Computation

### MATLAB (verify.m line 1104)
```matlab
ri_ = (ri - reshape(sum(Gxi,2),[n0 bSz]));
```

### Python (verify_helpers.py line 530-531)
```python
Gxi_sum = np.sum(Gxi, axis=1)  # Sum over generators: (n0, bSz)
ri_ = ri - Gxi_sum
```

**Both implementations are identical!**

### Why `ri_ ≈ 0`?

The splitting logic:
1. Creates generators `Gxi` from `ri` values
2. Sets `Gxi(dimIdx, genIdx, batchIdx) = ri(dimIdx, batchIdx)`
3. Computes `ri_ = ri - sum(Gxi, axis=1)`

**If `Gxi` contains generators that sum to `ri`, then `ri_ ≈ 0`!**

This happens when:
- `numInitGens >= n0`: A generator is created for each input dimension
- All `ri` values are placed into generators
- `sum(Gxi, axis=1) ≈ ri`, leaving `ri_ ≈ 0`

**This is expected behavior!** All uncertainty is encoded in generators, not in `ri_`.

## Key Question

**If `ri_ ≈ 0` is expected, why do generators collapse?**

The answer is NOT in `ri_`, but in how generators propagate through the network:
1. Generators `Gxi` should propagate through layers
2. But if generators become very small, they collapse to zero
3. This happens when `m = 0` in activation layers (when bounds collapse at `c ≤ 0`)

## Double Precision in Python

### Current State

NumPy uses **float64 (double precision)** by default for:
- Array creation: `np.array([1, 2, 3])` → float64
- Most operations: `np.sum()`, `np.dot()`, etc. → float64

### How to Ensure Double Precision

1. **Explicit dtype specification:**
   ```python
   x = np.array([1, 2, 3], dtype=np.float64)
   # or
   x = np.array([1, 2, 3], dtype='float64')
   ```

2. **Convert existing arrays:**
   ```python
   x = x.astype(np.float64)
   ```

3. **Set default dtype (not recommended, affects all arrays):**
   ```python
   np.set_printoptions(precision=15)  # For display only
   # No global dtype setting in NumPy
   ```

4. **Check current dtype:**
   ```python
   print(x.dtype)  # Should show 'float64'
   ```

### Where Precision Matters

1. **Network weights:**
   - ONNX models often use float32
   - Need to convert: `W = W.astype(np.float64)`

2. **Input data:**
   - Ensure inputs are float64: `x = x.astype(np.float64)`

3. **Intermediate computations:**
   - NumPy operations preserve dtype
   - But mixing float32 and float64 can cause issues

### MATLAB vs Python Precision

- **MATLAB**: Uses double precision (float64) by default
- **Python/NumPy**: Also uses float64 by default, BUT:
  - ONNX models often use float32
  - Some operations might cast to float32
  - Need to ensure all arrays are float64

### How to Check and Fix Precision Issues

1. **Add dtype checks:**
   ```python
   def ensure_float64(x):
       if x.dtype != np.float64:
           return x.astype(np.float64)
       return x
   ```

2. **Convert at network load:**
   ```python
   # In readONNXNetwork or verify
   for layer in nn.layers:
       if hasattr(layer, 'W'):
           layer.W = layer.W.astype(np.float64)
       if hasattr(layer, 'b'):
           layer.b = layer.b.astype(np.float64)
   ```

3. **Convert inputs:**
   ```python
   x = x.astype(np.float64)
   r = r.astype(np.float64)
   A = A.astype(np.float64)
   b = b.astype(np.float64)
   ```

## Recommendations

1. **Verify all arrays are float64:**
   - Add dtype checks at critical points
   - Convert ONNX weights to float64
   - Ensure inputs are float64

2. **Compare with MATLAB:**
   - Run MATLAB debug script to see if MATLAB also has `ri_ ≈ 0`
   - Compare `r` values in activation layers
   - Compare `m` values when bounds collapse

3. **Investigate generator collapse:**
   - The issue is NOT in `ri_ ≈ 0` (this is expected)
   - The issue is in how generators propagate through layers
   - Need to see why `m = 0` more often in Python

## Next Steps

1. **Run MATLAB debug script** (after fixing ONNX format)
2. **Compare `r` and `m` values** between MATLAB and Python
3. **Check if precision is the issue:**
   - Ensure all arrays are float64
   - Compare numerical values at each step
4. **If precision is not the issue:**
   - Investigate why bounds collapse faster in Python
   - Check if splitting produces different `Gxi` values

