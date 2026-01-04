# Hybrid Dynamics Dependency Check

## Overview
This document tracks all dependencies needed for hybridDynamics translation and their status.

## ✅ ContSet Classes - ALL VERIFIED

### ✅ Translated Classes
1. **polytope** - ✅ EXISTS (`cora_python/contSet/polytope/polytope.py`)
2. **zonotope** - ✅ EXISTS (`cora_python/contSet/zonotope/zonotope.py`)
3. **zonoBundle** - ✅ EXISTS (`cora_python/contSet/zonoBundle/zonoBundle.py`)
4. **conZonotope** - ✅ EXISTS (`cora_python/contSet/conZonotope/conZonotope.py`)
5. **polyZonotope** - ✅ EXISTS (`cora_python/contSet/polyZonotope/polyZonotope.py`)
6. **interval** - ✅ EXISTS (`cora_python/contSet/interval/interval.py`)
7. **levelSet** - ✅ EXISTS (`cora_python/contSet/levelSet/levelSet.py`)
8. **fullspace** - ✅ EXISTS (`cora_python/contSet/fullspace/fullspace.py`)
9. **conPolyZono** - ✅ EXISTS (`cora_python/contSet/conPolyZono/conPolyZono.py`) - NOT USED in hybridDynamics

## ✅ Methods Used in hybridDynamics - ALL VERIFIED

### From contSet classes:
1. **and_** - Intersection method
   - Used in: guardIntersect_polytope, guardIntersect_conZonotope, guardIntersect_levelSet
   - Status: ✅ VERIFIED - EXISTS in multiple contSet classes

2. **vertices** - Get vertices of set
   - Used in: guardIntersect_polytope
   - Status: ✅ VERIFIED - EXISTS in `contSet/contSet/vertices.py`

3. **interval.empty(dim)** - Create empty interval
   - Used in: guardIntersect_levelSet, guardIntersect_conZonotope, guardIntersect_zonoGirard
   - Status: ✅ VERIFIED - EXISTS in `contSet/interval/empty.py`

4. **interval.enclosePoints(V)** - Enclose points with interval
   - Used in: guardIntersect_polytope
   - Status: ✅ VERIFIED - EXISTS in `contSet/interval/enclosePoints.py`

5. **zonotope.enclosePoints(V, method)** - Enclose points with zonotope
   - Used in: guardIntersect_polytope
   - Status: ✅ VERIFIED - EXISTS in `contSet/zonotope/enclosePoints.py`

6. **reduce(set, technique, order)** - Reduce set order
   - Used in: guardIntersect_conZonotope, guardIntersect_hyperplaneMap
   - Status: ✅ VERIFIED - EXISTS in `contSet/zonotope/reduce.py`, `contSet/contSet/reduce.py`

7. **representsa_(set, type, tol)** - Check if set represents another type
   - Used in: guardIntersect, checkFlow, calcBasis
   - Status: ✅ VERIFIED - EXISTS in all relevant classes (polytope, zonotope, conZonotope, polyZonotope, interval, levelSet, etc.)

8. **contains_(set, point, method, tol)** - Check containment
   - Used in: guardIntersect_pancake
   - Status: ✅ VERIFIED - EXISTS in `contSet/polytope/contains_.py`, `contSet/zonotope/contains_.py`, etc.

9. **center(set)** - Get center of set
   - Used in: guardIntersect_pancake, guardIntersect_hyperplaneMap
   - Status: ✅ VERIFIED - EXISTS in `contSet/zonotope/center.py`, `contSet/polytope/center.py`, `contSet/conZonotope/center.py`, etc.

10. **polyZonotope(interval)** - Constructor from interval
    - Used in: guardIntersect_levelSet
    - Status: ✅ VERIFIED - Constructor accepts Interval objects

11. **conZonotope(zonotope)** - Constructor/conversion
    - Used in: guardIntersect_conZonotope, guardIntersect_zonoGirard
    - Status: ✅ VERIFIED - Constructor accepts Zonotope objects (line 134 in `conZonotope.py`)

12. **zonotope(polyZonotope)** - Conversion
    - Used in: guardIntersect_conZonotope, guardIntersect
    - Status: ✅ VERIFIED - EXISTS in `contSet/polyZonotope/zonotope.py`

13. **polytope(set)** - Conversion
    - Used in: guardIntersect_polytope (aux_conv2polytope)
    - Status: ✅ VERIFIED - Constructor accepts Zonotope objects (line 138-148 in `polytope.py`)

## ✅ ReachSet Class - VERIFIED

### Methods Used:
1. **R.timePoint.set** - Time-point reachable sets
   - Status: ✅ VERIFIED - Property exists in `ReachSet` class

2. **R.timeInterval.set** - Time-interval reachable sets
   - Status: ✅ VERIFIED - Property exists in `ReachSet` class

3. **R.timeInterval.time** - Time intervals
   - Status: ✅ VERIFIED - Property exists in `ReachSet` class

4. **updateTime(R, tStart)** - Update time in reachSet
   - Status: ✅ VERIFIED - EXISTS in `g/classes/reachSet/updateTime.py`

5. **check(spec, R)** - Check specification
   - Status: ✅ VERIFIED - EXISTS in `specification/specification/check.py`

**Location**: `cora_python/g/classes/reachSet/reachSet.py`

## ✅ ContDynamics Classes - VERIFIED

### Classes Used:
1. **linearSys** - Linear continuous dynamics
   - Status: ✅ VERIFIED - EXISTS in `cora_python/contDynamics/linearSys/linearSys.py`

2. **nonlinearSys** - Nonlinear continuous dynamics
   - Status: ✅ VERIFIED - EXISTS in `cora_python/contDynamics/nonlinearSys/nonlinearSys.py`

### Methods Used:
1. **reach(sys, params, options)** - Compute reachable set
   - Status: ✅ VERIFIED - EXISTS in `contDynamics/linearSys/reach.py` and `contDynamics/nonlinearSys/reach.py`

2. **simulate(sys, params, options)** - Simulate trajectories
   - Status: ✅ VERIFIED - EXISTS in `contDynamics/linearSys/simulate.py` and `contDynamics/nonlinearSys/simulate.py`

3. **getfcn(sys, params)** - Get function handle
   - Status: ✅ VERIFIED - EXISTS in `contDynamics/linearSys/getfcn.py` and `contDynamics/nonlinearSys/getfcn.py`

## ✅ Derive Function - TRANSLATED

### Status: ✅ TRANSLATED
- **MATLAB**: `cora_matlab/global/functions/verbose/write/derive.m`
- **Python**: ✅ `cora_python/g/functions/verbose/write/derive.py`
- **Used in**: `nonlinearReset.derivatives` (also ✅ translated)
- **Dependencies**: ✅ All dependencies exist

## ❌ NOT NEEDED

### Taylor Models (tylm)
- **Status**: NOT USED in hybridDynamics
- **Note**: The `derive` function in `nonlinearReset/derivatives.m` is for symbolic computation, not tylm
- **Conclusion**: tylm is NOT a dependency for hybridDynamics

## ✅ Verification Results - COMPLETE

### ✅ All Methods Verified
1. ✅ **Interval.empty(dim)** - EXISTS
2. ✅ **Interval.enclosePoints(V)** - EXISTS
3. ✅ **Zonotope.enclosePoints(V, method)** - EXISTS
4. ✅ **and_** - EXISTS
5. ✅ **vertices** - EXISTS
6. ✅ **PolyZonotope(interval)** - Constructor accepts Interval
7. ✅ **reduce(set, technique, order)** - EXISTS
8. ✅ **representsa_(set, type, tol)** - EXISTS in all relevant classes
9. ✅ **contains_(set, point, method, tol)** - EXISTS
10. ✅ **center(set)** - EXISTS in all relevant classes
11. ✅ **conZonotope(zonotope)** - Constructor accepts Zonotope
12. ✅ **zonotope(polyZonotope)** - Method exists
13. ✅ **polytope(set)** - Constructor accepts Zonotope

### ✅ All Classes Verified
1. ✅ **ReachSet** - EXISTS with all required properties and methods
2. ✅ **LinearSys** - EXISTS with all required methods
3. ✅ **NonlinearSys** - EXISTS with all required methods

## 🎉 Final Status

**ALL DEPENDENCIES FOR HYBRIDDYNAMICS TRANSLATION ARE PRESENT AND VERIFIED!**

Every single class, method, and functionality needed for the hybridDynamics translation has been verified to exist in the Python codebase. The translation can proceed with complete confidence.
