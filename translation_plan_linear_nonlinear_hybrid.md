# Translation Plan: linearSys, nonlinearSys, and hybridDynamics Functions

**Created:** 2025-12-XX  
**Status:** Planning Phase  
**Priority:** High

---

## Overview

This document provides a detailed translation plan for the following functions:

1. **linearSys**: `verifyFast` (option `suppFunc`) → `priv_verifyRA_supportFunc`
2. **nonlinearSys**: `initReach`, `post`, `contDynamics.reach` (derivatives.m calls), `initReach_adaptive` (optional)
3. **hybridDynamics**: `hybridAutomaton.reach` → `location.reach` → `guardIntersect` methods

---

## 1. LINEAR SYS: verifyFast (suppFunc option)

### 1.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/contDynamics/@linearSys/private/priv_verifyRA_supportFunc.m` (1470 lines)
- `cora_matlab/contDynamics/@linearSys/verify.m` (calls `priv_verifyRA_supportFunc`)

**Test Files:**
- `cora_matlab/examples/ARCHcompetition/linear/benchmark_linear_verifyFast_ARCH23_*.m` (multiple benchmark files)
- `cora_matlab/examples/contDynamics/linearSys/example_linear_verify.m`

**Dependencies:**
- `validateOptions` (linearSys)
- `aux_getSetsFromSpec` (auxiliary function in same file)
- `aux_canonicalForm` (auxiliary function in same file)
- `aux_initExpmat` (auxiliary function in same file)
- `aux_initStructsFlags` (auxiliary function in same file)
- `aux_timeStep` (auxiliary function in same file)
- `aux_intmat` (interval matrices computation)
- `aux_Pu`, `aux_underPU`, `aux_overPU` (particular solution computations)
- `aux_removeFromUnsat` (specification handling)
- `aux_affineTimeStep`, `aux_affinePUTimeStep` (time step adaptation)
- `aux_getApower`, `aux_getAposneg` (matrix power computations)
- `aux_robustProjection`, `aux_tightenSet` (set operations)
- `aux_enclosingInterval`, `aux_bound_intersect_2D`, `aux_dichotomicSearch` (2D intersection)
- `aux_lineIntersect2D` (line intersection)

**Input/Output Signature:**
```matlab
function [res,fals,savedata] = priv_verifyRA_supportFunc(linsys,params,options,spec)
```
- **Inputs:**
  - `linsys`: linearSys object
  - `params`: model parameters (R0, U, uTransVec, vTransVec, tu, tStart, tFinal)
  - `options`: algorithm parameters (verbose, normC, etc.)
  - `spec`: specification object (safe/unsafe sets)
- **Outputs:**
  - `res`: boolean (true if verified, false if falsified, -1 if undecided)
  - `fals`: struct with falsifying trajectory (x0, u, tu, tFinal)
  - `savedata`: struct with distances and computation time

**MATLAB-Specific Operations:**
1. Exponential matrix computation: `expm(A * timeStep)`
2. Interval arithmetic for curvature errors
3. Support function evaluation: `l'*c + sum(abs(l'*G),2)`
4. Taylor series for particular solutions (until floating-point precision)
5. Back-propagation of support function directions
6. Time step adaptation based on distance to unsafe set
7. Specification handling (safe sets, unsafe sets)

**Logic Breakdown:**
1. **Initialization:**
   - Validate options and parameters
   - Extract safe/unsafe sets from specifications
   - Convert to canonical form (Ax + Bu + c + w → Ax + u)
   - Initialize exponential matrix structures
   - Compute initial distance to unsafe set

2. **Main Loop (until verified/falsified):**
   - Compute time step size (adaptive)
   - Compute exponential matrix for time step
   - Compute interval matrices for curvature errors (F, G)
   - Propagate affine solution (back-propagate support function directions)
   - Compute particular solution (constant input)
   - Compute distance contributions:
     - `dist.affine_tp`: time-point affine solution distance
     - `dist.Cbloat`: curvature error distance
     - `dist.Pu`: particular solution distance
     - `dist.underPU`, `dist.overPU`: input set bounds (if U provided)
   - Check for falsification (any distance > 0)
   - Update verified time intervals
   - Adapt time step if needed

3. **Output:**
   - Return verification result
   - Return falsifying trajectory if found
   - Return saved data for visualization

**Edge Cases:**
- Initial set already intersects unsafe set
- Time step size becomes too small (< 1e-12)
- Exponential matrix computation doesn't converge
- Empty input set vs. constant input
- Multiple specifications

**Additional Requirements from Tests:**
- Must handle piecewise-constant input vectors
- Must handle time-varying output equation vectors
- Must support both safe sets and unsafe sets
- Must return falsifying trajectory on falsification

### 1.2 Manual Specifications

**Manual Section:** Verification Algorithms (Section X.X)

**Official Description:**
> The support function-based verification algorithm provides a faster alternative to standard reachability analysis by computing reachable sets implicitly with respect to their distance to unsafe sets. The algorithm uses support functions to evaluate distances without explicitly computing the full reachable set.

**Parameter Specifications:**
- `options.verifyAlg = 'reachavoid:supportFunc'` (required)
- `options.verbose`: boolean for verbose output
- `options.normC`: norm of output matrix C (for heuristic)

**Implementation Requirements:**
- Must handle specifications given as halfspaces (polytopes)
- Must support both safe sets and unsafe sets
- Must compute distances accurately using support functions
- Must adapt time step size based on distance to unsafe set

### 1.3 Python Implementation Plan

**Torch/NumPy Equivalents:**
- `expm(A * timeStep)` → `torch.linalg.matrix_exp(A * timeStep)` or `scipy.linalg.expm`
- Interval arithmetic → `cora_python.g.functions.helper.sets.contSet.interval`
- Support function → `cora_python.contSet.zonotope.supportFunc_`
- Matrix operations → `torch` or `numpy` (prefer torch for GPU support)

**Data Structure Changes:**
- MATLAB structs → Python dictionaries
- Cell arrays → Python lists
- MATLAB's `interval` → `cora_python.contSet.interval.Interval`
- MATLAB's `zonotope` → `cora_python.contSet.zonotope.Zonotope`

**Edge Case Handling:**
- Check for initial set intersection before main loop
- Handle time step size becoming too small
- Handle exponential matrix convergence failures
- Handle empty input sets vs. constant inputs
- Handle multiple specifications

**File Structure:**
```
cora_python/contDynamics/linearSys/
├── private/
│   ├── __init__.py
│   └── priv_verifyRA_supportFunc.py  (main function + all aux functions)
```

**Dependencies:**
- `cora_python.contSet.zonotope.Zonotope`
- `cora_python.contSet.interval.Interval`
- `cora_python.contSet.polytope.Polytope`
- `cora_python.contDynamics.linearSys.linearSys`
- `cora_python.g.functions.helper.*` (various helpers)

### 1.4 Test Implementation Plan

**MATLAB Test Cases Found:**
- Multiple ARCH competition benchmarks (heat3D, iss, beam)
- Example with safe/unsafe sets
- Tests with piecewise-constant inputs
- Tests with time-varying output equations

**Additional Edge Cases:**
- Empty input set
- Constant input over entire horizon
- Multiple specifications
- Initial set already unsafe
- Time step adaptation scenarios

**Verification Method:**
- Run MATLAB benchmarks and extract input/output pairs
- Compare Python results against MATLAB
- Use tolerance `atol=1e-6` for floating-point comparisons

---

## 2. NONLINEAR SYS: initReach

### 2.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/contDynamics/@nonlinearSys/initReach.m` (139 lines)

**Test Files:**
- `cora_matlab/unitTests/contDynamics/nonlinearSys/test_nonlinearSys_initReach.m` (if exists)
- Examples in `cora_matlab/examples/contDynamics/nonlinearSys/`

**Dependencies:**
- `linReach` (nonlinearSys method)
- `split` (set operation)
- `aux_initReach_linRem` (auxiliary function for linRem algorithm)

**Input/Output Signature:**
```matlab
function [Rnext,options] = initReach(nlnsys,Rinit,params,options)
```
- **Inputs:**
  - `nlnsys`: nonlinearSys object
  - `Rinit`: initial reachable set (can be cell array if split)
  - `params`: model parameters
  - `options`: algorithm settings (alg, maxError, etc.)
- **Outputs:**
  - `Rnext`: struct with `tp`, `ti`, `R0` (cell arrays)
  - `options`: updated options (may contain POpt)

**MATLAB-Specific Operations:**
1. Cell array handling for split sets
2. Recursive calls for split sets
3. Error field in reachable set structs

**Logic Breakdown:**
1. **Initialization:**
   - Convert single set to cell array if needed
   - Initialize error field if first time step

2. **Algorithm Selection:**
   - If `options.alg == 'linRem'`: call `aux_initReach_linRem`
   - Otherwise: proceed with standard algorithm

3. **Main Loop (over parallel sets):**
   - Call `linReach` for each set
   - Check if splitting needed (`dimForSplit` not empty)
   - If splitting:
     - Split initial set
     - Reset error fields
     - Recursively call `initReach` on split sets
   - If no splitting:
     - Store results (tp, ti, R0)
     - Set `prev` and `parent` fields

4. **Output:**
   - Return struct with `tp`, `ti`, `R0` cell arrays

**Edge Cases:**
- First time step (single set, not cell array)
- Sets requiring splitting
- Multiple parallel sets
- `linRem` algorithm path

### 2.2 Manual Specifications

**Manual Section:** Nonlinear System Reachability (Section X.X)

**Official Description:**
> `initReach` computes the reachable continuous set for the first time step of a nonlinear system. It handles initial set splitting if the linearization error is too large.

### 2.3 Python Implementation Plan

**Torch/NumPy Equivalents:**
- Cell arrays → Python lists
- Structs → Python dictionaries or dataclasses
- Set operations → CORA Python set classes

**Data Structure Changes:**
- MATLAB struct with fields → Python dict or dataclass
- Cell arrays → lists
- Error handling → Python exceptions

**File Structure:**
```
cora_python/contDynamics/nonlinearSys/
├── initReach.py
└── private/
    └── aux_initReach_linRem.py (if needed separately)
```

**Dependencies:**
- `cora_python.contDynamics.nonlinearSys.nonlinearSys`
- `cora_python.contDynamics.nonlinearSys.linReach` (must be translated first)
- `cora_python.contSet.*.split` (set splitting operation)

### 2.4 Test Implementation Plan

**MATLAB Test Cases:**
- Basic initialization
- Initial set splitting scenarios
- Multiple parallel sets
- `linRem` algorithm path

**Additional Edge Cases:**
- Very small initial sets
- Sets requiring multiple splits
- Sets with zero error

---

## 3. NONLINEAR SYS: post

### 3.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/contDynamics/@nonlinearSys/post.m` (67 lines)

**Dependencies:**
- `initReach` (nonlinearSys method)
- `reduce` (set reduction)
- `deleteRedundantSets` (auxiliary function)
- `restructure` (polyZonotope operation)
- `approxVolumeRatio` (polyZonotope operation)

**Input/Output Signature:**
```matlab
function [Rnext,options] = post(nlnsys,R,params,options)
```
- **Inputs:**
  - `nlnsys`: nonlinearSys object
  - `R`: reachable set from previous time step (struct with `tp`, `ti`)
  - `params`: model parameters
  - `options`: algorithm settings
- **Outputs:**
  - `Rnext`: reachable set for next time step
  - `options`: updated options

**Logic Breakdown:**
1. **Polynomial Zonotope Restructuring** (if `alg == 'poly'`):
   - Check volume ratio
   - Restructure if ratio > `maxPolyZonoRatio`

2. **Reachability Computation:**
   - Call `initReach` (nonlinear systems need constant re-initialization)

3. **Set Reduction:**
   - Reduce zonotopes using specified technique

4. **Redundant Set Removal:**
   - Delete redundant reachable sets

**Edge Cases:**
- Polynomial zonotope restructuring
- Empty sets
- Sets requiring reduction

### 3.2 Python Implementation Plan

**File Structure:**
```
cora_python/contDynamics/nonlinearSys/
└── post.py
```

**Dependencies:**
- `cora_python.contDynamics.nonlinearSys.initReach`
- `cora_python.contSet.polyZonotope.PolyZonotope` (restructure, approxVolumeRatio)
- `cora_python.contSet.zonotope.Zonotope` (reduce)
- `deleteRedundantSets` (auxiliary function, may need translation)

---

## 4. NONLINEAR SYS: contDynamics.reach (derivatives.m calls)

### 4.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/contDynamics/@contDynamics/reach.m` (214 lines)
- `cora_matlab/contDynamics/@contDynamics/derivatives.m` (483 lines)

**Key Call:**
```matlab
% Line 61 in reach.m:
derivatives(sys,options);
```

**Dependencies:**
- `derivatives` computes symbolic derivatives (Jacobians, Hessians, tensors)
- Generates MATLAB files for derivative evaluation
- Uses symbolic math toolbox

**Logic Breakdown:**
1. **Symbolic Variable Creation:**
   - Create symbolic variables for states, inputs, outputs, parameters

2. **Derivative Computation:**
   - Jacobians (first-order)
   - Hessians (second-order)
   - Third-order tensors
   - Higher-order tensors (if `tensorOrder >= 4`)

3. **File Generation:**
   - Write MATLAB files for derivative evaluation
   - Store in `models/auxiliary/{sys.name}/`

**Python Implementation Considerations:**
- Python doesn't have symbolic math like MATLAB
- Options:
  1. Use `sympy` for symbolic computation (slower, but exact)
  2. Use automatic differentiation (PyTorch, JAX)
  3. Pre-compute derivatives in MATLAB and load in Python
  4. Use numerical differentiation (not recommended)

**Recommended Approach:**
- Use PyTorch's automatic differentiation for runtime computation
- For pre-computation, use `sympy` to generate Python functions
- Store generated functions in `cora_python/models/auxiliary/{sys.name}/`

### 4.2 Python Implementation Plan

**File Structure:**
```
cora_python/contDynamics/contDynamics/
├── derivatives.py  (main function)
└── private/
    ├── aux_defaultOptions.py
    ├── aux_insertSymVariables.py
    ├── aux_jacobians.py
    ├── aux_hessians.py
    ├── aux_thirdOrderDerivatives.py
    └── writeHessianTensorFile.py (or similar for Python)
```

**Dependencies:**
- `sympy` for symbolic computation (optional, for pre-generation)
- `torch` for automatic differentiation (runtime)
- File I/O for storing generated functions

**Implementation Strategy:**
1. **Runtime Computation (Primary):**
   - Use PyTorch's `torch.autograd` for automatic differentiation
   - Compute derivatives on-the-fly during reachability analysis
   - Cache results if possible

2. **Pre-computation (Optional):**
   - Use `sympy` to generate Python functions
   - Store in `cora_python/models/auxiliary/{sys.name}/`
   - Load and use during reachability analysis

---

## 5. NONLINEAR SYS: priv_abstrerr_lin

### 5.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/contDynamics/@contDynamics/private/priv_abstrerr_lin.m` (216 lines)

**Dependencies:**
- `setHessian` (nonlinearSys method)
- `setThirdOrderTensor` (nonlinearSys method)
- `interval` (set operation)
- `quadMap` (set operation)
- `cubMap` (set operation, for tensorOrder == 3)
- `initRangeBoundingObjects` (for Taylor models/zoo)

**Input/Output Signature:**
```matlab
function [trueError,VerrorDyn] = priv_abstrerr_lin(sys,R,params,options)
```
- **Inputs:**
  - `sys`: nonlinearSys or nonlinParamSys object
  - `R`: reachable set (time-interval solution)
  - `params`: model parameters
  - `options`: options struct (tensorOrder, lagrangeRem, etc.)
- **Outputs:**
  - `trueError`: abstraction error (interval)
  - `VerrorDyn`: abstraction error (zonotope)

**Logic Breakdown:**
1. **Compute Intervals:**
   - Interval of reachable set: `IHx = interval(R)`
   - Total interval: `totalInt_x = IHx + sys.linError.p.x`
   - Input interval: `IHu = interval(params.U)`
   - Total input interval: `totalInt_u = IHu + sys.linError.p.u`

2. **Tensor Order 2:**
   - Set Hessian to interval mode
   - Evaluate Hessian matrix
   - Compute Lagrange remainder: `0.5 * dz' * H_ * dz`
   - Return error as interval and zonotope

3. **Tensor Order 3:**
   - Set Hessian to standard mode
   - Set third-order tensor to interval mode
   - Evaluate Hessian and third-order tensor
   - Compute second-order error: `0.5 * quadMap(Z, H)`
   - Compute third-order error: `1/6 * error_sum` (using tensor)
   - Combine errors

**Edge Cases:**
- Empty tensor entries
- Interval arithmetic vs. Taylor models/zoo
- Parameter systems (nonlinParamSys)

### 5.2 Python Implementation Plan

**File Structure:**
```
cora_python/contDynamics/contDynamics/
└── private/
    └── priv_abstrerr_lin.py
```

**Dependencies:**
- `cora_python.contSet.interval.Interval`
- `cora_python.contSet.zonotope.Zonotope`
- `cora_python.g.functions.helper.sets.contSet.zonotope.quadMap`
- `cora_python.g.functions.helper.sets.contSet.zonotope.cubMap` (if exists)

---

## 6. NONLINEAR SYS: priv_abstrerr_poly

### 6.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/contDynamics/@contDynamics/private/priv_abstrerr_poly.m` (173 lines)

**Dependencies:**
- `priv_precompStatError` (pre-computed static error)
- `setHessian`, `setThirdOrderTensor` (nonlinearSys methods)
- `quadMap`, `cubMap` (set operations)
- `reduce` (set reduction)

**Input/Output Signature:**
```matlab
function [trueError, VerrorDyn, VerrorStat] = priv_abstrerr_poly(sys, ...
    Rall, Rdiff, params, options, H, Zdelta, VerrorStat, T, ind3, Zdelta3)
```
- **Inputs:**
  - `sys`: nonlinearSys object
  - `Rall`: time-interval reachable set
  - `Rdiff`: difference between reachable sets
  - `params`: model parameters
  - `options`: options struct
  - `H`, `Zdelta`, `VerrorStat`, `T`, `ind3`, `Zdelta3`: from `priv_precompStatError`
- **Outputs:**
  - `trueError`: overall linearization error (interval)
  - `VerrorDyn`: dynamic linearization error (zonotope)
  - `VerrorStat`: static linearization error (zonotope, updated)

**Logic Breakdown:**
1. **Compute Intervals:**
   - Interval of reachable set and input
   - Translate by linearization point

2. **Second-Order Error:**
   - `error_secondOrder_dyn = 0.5*(quadMap(Zdelta,Z_diff,H) + ...)`

3. **Third-Order Error:**
   - If `tensorOrder == 3`: evaluate tensor with intervals
   - If `tensorOrder >= 4`: use `cubMap` with reduced sets

4. **Higher-Order Terms:**
   - Evaluate intermediate Taylor terms (if `tensorOrder >= 4`)
   - Compute Lagrange remainder

5. **Combine Results:**
   - `VerrorDyn = error_secondOrder_dyn + error_thirdOrder_dyn + remainder`
   - `trueError = supremum(abs(interval(VerrorDyn) + interval(VerrorStat)))`

### 6.2 Python Implementation Plan

**File Structure:**
```
cora_python/contDynamics/contDynamics/
└── private/
    └── priv_abstrerr_poly.py
```

**Dependencies:**
- `priv_precompStatError` (must be translated first)
- `quadMap`, `cubMap` (set operations)
- `reduce` (set reduction)

---

## 7. NONLINEAR SYS: priv_precompStatError

### 7.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/contDynamics/@contDynamics/private/priv_precompStatError.m` (118 lines)

**Dependencies:**
- `setHessian`, `setThirdOrderTensor` (nonlinearSys methods)
- `reduce` (set reduction)
- `cartProd` (set operation)
- `quadMap`, `cubMap` (set operations)

**Input/Output Signature:**
```matlab
function [H,Zdelta,errorStat,T,ind3,Zdelta3] = priv_precompStatError(sys,Rdelta,params,options)
```
- **Inputs:**
  - `sys`: nonlinearSys object
  - `Rdelta`: shifted reachable set at beginning of time step
  - `params`: model parameters
  - `options`: options struct
- **Outputs:**
  - `H`: Hessian matrix
  - `Zdelta`: zonotope over-approximating reachable set + input set
  - `errorStat`: static linearization error
  - `T`: third-order tensor (if `tensorOrder >= 4`)
  - `ind3`: indices of non-zero tensor entries
  - `Zdelta3`: reduced set for third-order tensor evaluation

**Logic Breakdown:**
1. **Set Reduction:**
   - Reduce `Rdelta` to specified order

2. **Cartesian Product:**
   - `Z = cartProd(Rred, Ustat)` (Ustat is zero zonotope)
   - `Zdelta = cartProd(Rdelta, Ustat)`

3. **Hessian Computation:**
   - Call `sys.hessian` at linearization point

4. **Static Second-Order Error:**
   - `errorSecOrdStat = 0.5*quadMap(Z, H)`

5. **Third-Order Error** (if `tensorOrder >= 4`):
   - Reduce sets further (if `errorOrder3` specified)
   - Compute third-order tensor
   - `errorThirdOrdStat = 1/6 * cubMap(Z,T,ind3)`

6. **Combine Errors:**
   - `errorStat = errorSecOrdStat + errorThirdOrdStat` (or exactPlus for polyZonotope)

7. **Reduce Error Set:**
   - Reduce `errorStat` to intermediate order

### 7.2 Python Implementation Plan

**File Structure:**
```
cora_python/contDynamics/contDynamics/
└── private/
    └── priv_precompStatError.py
```

**Dependencies:**
- `cora_python.contSet.zonotope.Zonotope`
- `cora_python.g.functions.helper.sets.contSet.zonotope.cartProd`
- `cora_python.g.functions.helper.sets.contSet.zonotope.quadMap`
- `cora_python.g.functions.helper.sets.contSet.zonotope.cubMap`

---

## 8. NONLINEAR SYS: initReach_adaptive (OPTIONAL)

### 8.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/contDynamics/@nonlinearSys/initReach_adaptive.m` (37 lines)

**Dependencies:**
- `linReach_adaptive` (nonlinearSys method)

**Input/Output Signature:**
```matlab
function [Rnext,options] = initReach_adaptive(nlnsys,options)
```
- **Inputs:**
  - `nlnsys`: nonlinearSys object
  - `options`: options struct (must contain `options.R`)
- **Outputs:**
  - `Rnext`: reachable set struct
  - `options`: updated options

**Logic Breakdown:**
1. Call `linReach_adaptive` with `options.R`
2. Store results in `Rnext` struct

**Note:** This is a simple wrapper around `linReach_adaptive`. If `linReach_adaptive` is not translated, this can be skipped.

### 8.2 Python Implementation Plan

**File Structure:**
```
cora_python/contDynamics/nonlinearSys/
└── initReach_adaptive.py
```

**Dependencies:**
- `cora_python.contDynamics.nonlinearSys.linReach_adaptive` (must be translated first)

---

## 9. HYBRID DYNAMICS: hybridAutomaton.reach

### 9.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/hybridDynamics/@hybridAutomaton/reach.m` (368 lines)

**Dependencies:**
- `validateOptions` (hybridAutomaton)
- `priv_flowDerivatives` (derivatives for flow in each location)
- `location.reach` (main computation)
- `derivatives` (for reset functions)
- `aux_check_flatHA_specification` (specification handling)
- `aux_outputSet` (output set computation)
- Various verbose display functions

**Input/Output Signature:**
```matlab
function [R,res] = reach(HA,params,options,varargin)
```
- **Inputs:**
  - `HA`: hybridAutomaton object
  - `params`: parameters (R0, startLoc, Uloc, uloc, Wloc, Vloc, tStart, tFinal)
  - `options`: options struct
  - `spec`: (optional) specification object
- **Outputs:**
  - `R`: reachSet object (array)
  - `res`: boolean (specification satisfaction)

**Logic Breakdown:**
1. **Initialization:**
   - Validate options
   - Compute derivatives for each location's flow
   - Check specifications

2. **Queue Initialization:**
   - Initialize queue with initial set, location, time, parent

3. **Main Loop (while queue not empty):**
   - Get first element from queue
   - Check for instant transitions
   - If instant transition:
     - Save reachable set
     - Compute reset
     - Add to queue
   - Else:
     - Compute derivatives for reset functions
     - Call `location.reach`
     - Process guard intersections (`Rjump`)
     - Add new branches to queue
     - Compute output sets
     - Store reachable sets

4. **Output:**
   - Return array of reachSet objects
   - Return specification satisfaction result

**Edge Cases:**
- Instant transitions
- Multiple locations
- Guard intersections
- Specification violations
- Empty queue

### 9.2 Python Implementation Plan

**File Structure:**
```
cora_python/hybridDynamics/hybridAutomaton/
├── reach.py
└── private/
    ├── priv_flowDerivatives.py
    ├── aux_check_flatHA_specification.py
    └── aux_outputSet.py
```

**Dependencies:**
- `cora_python.hybridDynamics.location.Location` (must have `reach` method)
- `cora_python.hybridDynamics.hybridAutomaton.HybridAutomaton`
- `cora_python.contDynamics.contDynamics.derivatives` (for reset functions)

---

## 10. HYBRID DYNAMICS: location.reach

### 10.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/hybridDynamics/@location/reach.m` (114 lines)

**Dependencies:**
- `contDynamics.reach` (for continuous evolution)
- `potInt` (potential intersection detection)
- `guardIntersect` (guard intersection computation)
- `evaluate` (reset function evaluation)
- `potOut` (remove parts outside invariant)
- `updateTime` (time update)
- `check` (specification checking)
- `aux_adaptSpecs` (specification adaptation)

**Input/Output Signature:**
```matlab
function [R,Rjump_,res] = reach(loc,params,options)
```
- **Inputs:**
  - `loc`: location object
  - `params`: parameters
  - `options`: options struct
- **Outputs:**
  - `R`: reachable set (reachSet object)
  - `Rjump_`: guard intersections (struct array)
  - `res`: boolean (specification satisfaction)

**Logic Breakdown:**
1. **Initialization:**
   - Initialize `Rjump` struct
   - Adapt specifications
   - Set `compOutputSet = false`

2. **Continuous Reachability:**
   - Call `reach(loc.contDynamics, params, options, specReach)`

3. **Guard Intersection Detection:**
   - Loop over reachable sets
   - Call `potInt` to detect guard intersections
   - Call `guardIntersect` to compute intersections
   - Compute reset and target location
   - Store in `Rjump_`

4. **Invariant Handling:**
   - Remove parts outside invariant (if `intersectInvariant`)

5. **Specification Checking:**
   - Check specifications at end of location

### 10.2 Python Implementation Plan

**File Structure:**
```
cora_python/hybridDynamics/location/
├── reach.py
└── private/
    └── aux_adaptSpecs.py
```

**Dependencies:**
- `cora_python.hybridDynamics.location.potInt`
- `cora_python.hybridDynamics.location.guardIntersect`
- `cora_python.hybridDynamics.location.potOut`
- `cora_python.hybridDynamics.location.updateTime`
- `cora_python.contDynamics.contDynamics.reach`

---

## 11. HYBRID DYNAMICS: guardIntersect Methods

### 11.1 MATLAB Source Analysis

**File Location:**
- `cora_matlab/hybridDynamics/@location/guardIntersect.m` (282 lines)
- `cora_matlab/hybridDynamics/@location/guardIntersect_zonoGirard.m` (404 lines)
- `cora_matlab/hybridDynamics/@location/guardIntersect_nondetGuard.m` (62 lines)
- `cora_matlab/hybridDynamics/@location/guardIntersect_levelSet.m` (58 lines)
- `cora_matlab/hybridDynamics/@location/guardIntersect_polytope.m` (122 lines)
- `cora_matlab/hybridDynamics/@location/guardIntersect_conZonotope.m` (71 lines)

**Main Function: `guardIntersect`**
- Dispatches to specific methods based on `options.guardIntersect`
- Handles grouping of sets
- Removes empty intersections
- Converts back to polynomial zonotopes if needed

**Method Priority (by importance):**
1. **guardIntersect_zonoGirard** (most important)
   - Zonotope-hyperplane intersection (Girard method)
   - Uses 2D intersection algorithm
   - Requires basis computation
   - Handles constrained hyperplanes

2. **guardIntersect_nondetGuard** (important)
   - For non-deterministic guards with large uncertainty
   - Uses basis transformation
   - Converts to constrained zonotopes

3. **guardIntersect_levelSet** (important)
   - For level set guards
   - Uses polynomial zonotopes
   - Domain tightening

4. **guardIntersect_polytope** (important)
   - Polytope-based method
   - Uses vertex computation
   - Supports 'box', 'pca', 'flow' enclosure methods

5. **guardIntersect_conZonotope** (less important)
   - Constrained zonotope method
   - Similar to zonoGirard but with constrained zonotopes

**Dependencies:**
- `calcBasis` (basis computation)
- `checkFlow` (flow direction checking)
- `aux_groupSets` (set grouping)
- `aux_removeEmptySets` (empty set removal)
- `aux_getInitialSet` (initial set retrieval)
- Various set operations (and_, vertices, etc.)

### 11.2 Python Implementation Plan

**File Structure:**
```
cora_python/hybridDynamics/location/
├── guardIntersect.py  (main dispatcher)
├── guardIntersect_zonoGirard.py
├── guardIntersect_nondetGuard.py
├── guardIntersect_levelSet.py
├── guardIntersect_polytope.py
├── guardIntersect_conZonotope.py
└── private/
    ├── aux_groupSets.py
    ├── aux_removeEmptySets.py
    ├── aux_getInitialSet.py
    └── (other aux functions)
```

**Implementation Priority:**
1. `guardIntersect_zonoGirard` (MUST)
2. `guardIntersect_nondetGuard` (SHOULD)
3. `guardIntersect_levelSet` (SHOULD)
4. `guardIntersect_polytope` (SHOULD)
5. `guardIntersect_conZonotope` (OPTIONAL)

**Dependencies:**
- `cora_python.contSet.zonotope.Zonotope`
- `cora_python.contSet.polytope.Polytope`
- `cora_python.contSet.levelSet.LevelSet`
- `cora_python.contSet.conZonotope.ConZonotope`
- `cora_python.contSet.zonoBundle.ZonoBundle`
- `calcBasis` (must be translated)

---

## 12. DEPENDENCY ORDER

### Translation Order (by dependencies):

**Phase 1: Foundation**
1. `priv_precompStatError` (used by `priv_abstrerr_poly`)
2. `priv_abstrerr_lin` (used by `initReach`)
3. `priv_abstrerr_poly` (used by `initReach`)

**Phase 2: Nonlinear Core**
4. `nonlinearSys.initReach` (used by `post` and `reach`)
5. `nonlinearSys.post` (used by `reach`)
6. `contDynamics.derivatives` (used by `reach`)
7. `nonlinearSys.initReach_adaptive` (optional, used by adaptive reach)

**Phase 3: Linear Verification**
8. `linearSys.priv_verifyRA_supportFunc` (standalone, but uses many helpers)

**Phase 4: Hybrid Foundation**
9. `location.guardIntersect` (main dispatcher)
10. `location.guardIntersect_zonoGirard` (priority 1)
11. `location.guardIntersect_nondetGuard` (priority 2)
12. `location.guardIntersect_levelSet` (priority 3)
13. `location.guardIntersect_polytope` (priority 4)
14. `location.guardIntersect_conZonotope` (priority 5, optional)
15. `location.reach` (uses `guardIntersect`)
16. `hybridAutomaton.reach` (uses `location.reach`)

**Phase 5: Optional**
17. `linearSys.priv_reach_krylov` (optional)

---

## 13. TESTING STRATEGY

### For Each Function:

1. **Unit Tests:**
   - Test each function in isolation
   - Use MATLAB-generated input/output pairs
   - Compare results with tolerance `atol=1e-6`

2. **Integration Tests:**
   - Test function chains (e.g., `initReach` → `post` → `reach`)
   - Test with real examples from MATLAB

3. **Benchmark Tests:**
   - Run ARCH competition benchmarks
   - Compare computation times (should be similar to MATLAB)

4. **Edge Case Tests:**
   - Empty sets
   - Degenerate cases
   - Boundary conditions

---

## 14. ESTIMATED COMPLEXITY

| Function | Lines (MATLAB) | Estimated Complexity | Priority |
|----------|----------------|---------------------|----------|
| `priv_verifyRA_supportFunc` | 1470 | Very High | High |
| `initReach` | 139 | Medium | High |
| `post` | 67 | Low | High |
| `derivatives` | 483 | High | High |
| `priv_abstrerr_lin` | 216 | Medium | High |
| `priv_abstrerr_poly` | 173 | Medium | High |
| `priv_precompStatError` | 118 | Medium | High |
| `initReach_adaptive` | 37 | Low | Low (optional) |
| `hybridAutomaton.reach` | 368 | High | High |
| `location.reach` | 114 | Medium | High |
| `guardIntersect` | 282 | High | High |
| `guardIntersect_zonoGirard` | 404 | High | High |
| `guardIntersect_nondetGuard` | 62 | Low | Medium |
| `guardIntersect_levelSet` | 58 | Low | Medium |
| `guardIntersect_polytope` | 122 | Medium | Medium |
| `guardIntersect_conZonotope` | 71 | Low | Low (optional) |
| `priv_reach_krylov` | 257 | High | Low (optional) |

**Total Estimated Lines:** ~4200+ lines of MATLAB code

---

## 15. NOTES AND CONSIDERATIONS

1. **Symbolic Computation:**
   - MATLAB uses symbolic math toolbox
   - Python should use `sympy` for pre-generation or PyTorch for runtime AD
   - Consider caching strategies

2. **Performance:**
   - `priv_verifyRA_supportFunc` is performance-critical
   - Use PyTorch for GPU acceleration where possible
   - Profile and optimize hot paths

3. **Numerical Stability:**
   - Exponential matrix computations require careful handling
   - Use appropriate tolerances for floating-point comparisons
   - Handle edge cases (e.g., singular matrices)

4. **Set Operations:**
   - Many functions depend on set operations (intersection, reduction, etc.)
   - Ensure all required set operations are translated first

5. **Testing:**
   - Generate MATLAB input/output pairs for validation
   - Use ARCH competition benchmarks as integration tests
   - Test edge cases thoroughly

---

## 16. NEXT STEPS


1. **Begin Translation in Dependency Order**
   - Start with Phase 1 (Foundation)
   - Progress through phases sequentially
2. **Continuous Testing**
   - Test each function as it's translated
   - Fix issues before moving to next function
3. **Documentation**
   - Update documentation as functions are translated
   - Document any deviations from MATLAB behavior

---

**End of Translation Plan**

