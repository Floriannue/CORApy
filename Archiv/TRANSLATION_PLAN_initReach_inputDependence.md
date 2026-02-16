# Translation Plan: initReach_inputDependence and Dependencies

**Status:** In Progress  
**Priority:** High (blocks `priv_select` tests)

---

## Overview

`initReach_inputDependence` is called from `linReach.py` when:
- `sys` is a `nonlinParamSys` AND
- `params.paramInt` is an `Interval`

This function and all its dependencies need to be translated.

---

## Dependencies Tree

### Main Function
- `initReach_inputDependence` (linParamSys method)

### Private Functions (called by initReach_inputDependence)
1. `priv_mappingMatrix` - computes mapping matrices
2. `priv_highOrderMappingMatrix` - computes high-order mapping matrices  
3. `priv_tie` - computes time interval error
4. `priv_inputSolution` - computes input solution
5. `priv_dependentHomSol` - computes dependent homogeneous solution
6. `priv_inputTie` - computes input time interval error (called by priv_inputSolution)

### External Dependencies
- `expmOneParam` - exponential matrix for one parameter (in `g/functions`)
- `expmMixed` - mixed exponential matrix computation (in `g/functions`)
- `expmIndMixed` - independent mixed exponential matrix (in `g/functions`)
- `intervalMatrix` - ✅ EXISTS
- `matZonotope` - ✅ EXISTS
- `enclose` - ✅ EXISTS (zonotope method)
- `reduce` - ✅ EXISTS (zonotope method)
- `LinearParamSys` class - ❌ MISSING (needs to be created)

---

## Translation Order

1. **Create LinearParamSys class** (base class structure)
2. **Translate expm functions** (if they don't exist)
3. **Translate private functions** (in dependency order):
   - `priv_mappingMatrix`
   - `priv_highOrderMappingMatrix`
   - `priv_tie`
   - `priv_inputTie`
   - `priv_inputSolution`
   - `priv_dependentHomSol`
4. **Translate main function**: `initReach_inputDependence`
5. **Create tests** with MATLAB I/O pairs

---

## Files to Create

```
cora_python/contDynamics/linearParamSys/
├── __init__.py
├── linearParamSys.py (class definition)
├── initReach_inputDependence.py
├── initReach.py (also needed by linReach)
├── errorSolution.py (also needed by linReach)
└── private/
    ├── __init__.py
    ├── priv_mappingMatrix.py
    ├── priv_highOrderMappingMatrix.py
    ├── priv_tie.py
    ├── priv_inputTie.py
    ├── priv_inputSolution.py
    └── priv_dependentHomSol.py
```

---

## Status

- [ ] LinearParamSys class
- [ ] expm functions (check if exist)
- [ ] priv_mappingMatrix
- [ ] priv_highOrderMappingMatrix
- [ ] priv_tie
- [ ] priv_inputTie
- [ ] priv_inputSolution
- [ ] priv_dependentHomSol
- [ ] initReach_inputDependence
- [ ] Tests

---

**Next Steps:** Start with LinearParamSys class, then translate functions in dependency order.
