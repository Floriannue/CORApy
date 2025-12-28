"""
checkAllContainments - given a set representation S, checks whether the
containment check S' < S can be performed for every other set representation S'
for which the check is implemented (only for sets in R^2).

The set representations 'affine', 'contSet', 'probZonotope', and 'zoo' are *not* tested.
Furthermore, this script does *not* check the formal correctness of the 'approx' method,
or that of any algorithm in additionalAlgorithms; it only verifies that they don't throw
any Python/CORA errors.

* This test tests A LOT. A LOT A LOT. Thus, requires a very long time.
In order to reduce the time and as the CI runs all the time, we opted to randomly
disable sub-tests that will, over time, all get tested.

Authors: MATLAB: Adrian Kulmburg
         Python: AI Assistant
Written: 12-July-2024 (MATLAB)
Last update: 22-May-2025 (MATLAB, randomly disabled tests*)
"""

import numpy as np
import random
from typing import List, Optional, Any


def checkAllContainments(S, Sdeg, Sempty, implementedSets, setsNonExact, 
                        additionalAlgorithms, additionalAlgorithms_specificSets):
    """
    Check all containment combinations for a given set type.
    
    Args:
        S: Basic set of the given type; should more or less be as big as a unit hypercube.
        Sdeg: Same as S, but with a degenerate version of the set (or an empty object
              if the set can not be degenerate, by construction).
        Sempty: Empty instance of the set
        implementedSets: List of strings describing all set representations S' for which
                         the containment S' < S is implemented (beyond point-containment,
                         emptyset, and fullspace)
        setsNonExact: List of strings describing all set representations from
                      implementedSets for which there is no 'exact' algorithm.
        additionalAlgorithms: List of strings describing all implemented algorithms,
                             beyond 'exact' and 'approx'
        additionalAlgorithms_specificSets: For each entry in additionalAlgorithms, you may
                                         specify on which set representations to apply the
                                         algorithm in question.
    """
    # Set global tolerance for every containment check
    tol = 1e-5
    
    # Import all ContSet types
    from cora_python.contSet.capsule.capsule import Capsule
    from cora_python.contSet.interval.interval import Interval
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
    from cora_python.contSet.polytope.polytope import Polytope
    from cora_python.contSet.zonotope.zonotope import Zonotope
    from cora_python.contSet.emptySet.emptySet import EmptySet
    from cora_python.contSet.fullspace.fullspace import Fullspace
    from cora_python.contSet.conPolyZono.conPolyZono import ConPolyZono
    from cora_python.contSet.conZonotope.conZonotope import ConZonotope
    from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle
    from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
    from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow
    from cora_python.contSet.levelSet.levelSet import LevelSet
    
    # Points
    point_contained = np.array([[0], [0]])  # Origin should be contained in basic set
    point_notContained = np.array([[100], [0]])  # A far away point that is not contained
    
    # Other compact sets
    # List of 'small' sets that are certainly contained within the basic set
    C_small = Capsule(np.array([[0], [0]]), np.array([[0.01], [0]]), 0.01)
    I_small = Interval(np.array([-0.01, -0.01]), np.array([0.01, 0.01]))
    cPZ_small = ConPolyZono(I_small)
    cZ_small = ConZonotope(I_small)
    E_small = Ellipsoid(0.001 * np.eye(2))
    # LevelSet: MATLAB uses syms x y; eqs = x^2 + y^2 - 0.001; ls_small = levelSet(eqs,[x;y],'<=')
    # Python: LevelSet creation with symbolic equations - skip for now as it requires symbolic math
    # ls_small = LevelSet(...)  # TODO: Implement symbolic level set creation
    P_small = 0.01 * Polytope(np.array([[1, 1, -1, -1], [1, -1, 1, -1]]))
    pZ_small = PolyZonotope(I_small)
    SpS_small = SpectraShadow(I_small)
    zB_small = ZonoBundle(I_small)
    Z_small = Zonotope(I_small)
    
    smallSets = [C_small, I_small, cPZ_small, cZ_small, E_small, 
                 P_small, pZ_small, SpS_small, zB_small, Z_small]
    # Note: ls_small (LevelSet) is commented out as it requires symbolic math
    
    # List of 'big' sets that are certainly not contained within the basic set
    C_big = Capsule(np.array([[0], [0]]), np.array([[100], [0]]), 100)
    I_big = Interval(np.array([-100, -100]), np.array([100, 100]))
    cPZ_big = ConPolyZono(I_big)
    cZ_big = ConZonotope(I_big)
    E_big = Ellipsoid(100 * np.eye(2))
    # LevelSet: MATLAB uses syms x y; eqs = x^2 + y^2 - 100; ls_big = levelSet(eqs,[x;y],'<=')
    # ls_big = LevelSet(...)  # TODO: Implement symbolic level set creation
    P_big = 100 * Polytope(np.array([[1, 1, -1, -1], [1, -1, 1, -1]]))
    pZ_big = PolyZonotope(I_big)
    SpS_big = SpectraShadow(I_big)
    zB_big = ZonoBundle(I_big)
    Z_big = Zonotope(I_big)
    
    bigSets = [C_big, I_big, cPZ_big, cZ_big, E_big, 
               P_big, pZ_big, SpS_big, zB_big, Z_big]
    # Note: ls_big (LevelSet) is commented out as it requires symbolic math
    
    # List of 'small' **degenerate** sets that are certainly contained
    C_degSmall = Capsule.empty(2)  # capsules can not be degenerate otherwise
    I_degSmall = Interval(np.array([-0.01, 0]), np.array([0.01, 0]))
    cPZ_degSmall = ConPolyZono(I_degSmall)
    cZ_degSmall = ConZonotope(I_degSmall)
    # E_degSmall = [1 0; 0 0] * ellipsoid(0.001*eye(2))
    # MATLAB: E_degSmall = [1 0; 0 0] * ellipsoid(0.001*eye(2))
    # Python: Use @ operator (matrix multiplication) - Ellipsoid should support __rmatmul__
    E_degSmall = np.array([[1, 0], [0, 0]]) @ Ellipsoid(0.001 * np.eye(2))
    # LevelSet: MATLAB uses syms x y; eqs = [x^2 + y^2 - 0.001; y; -y]; ls_degSmall = levelSet(eqs,[x;y],'<=')
    # ls_degSmall = LevelSet(...)  # TODO: Implement symbolic level set creation
    P_degSmall = 0.01 * Polytope(np.array([[-1, 1], [0, 0]]))
    pZ_degSmall = PolyZonotope(I_degSmall)
    SpS_degSmall = SpectraShadow(I_degSmall)
    zB_degSmall = ZonoBundle(I_degSmall)
    Z_degSmall = Zonotope(I_degSmall)
    
    degSmallSets = [C_degSmall, I_degSmall, cPZ_degSmall, cZ_degSmall, E_degSmall,
                    P_degSmall, pZ_degSmall, SpS_degSmall, zB_degSmall, Z_degSmall]
    # Note: ls_degSmall (LevelSet) is commented out as it requires symbolic math
    
    # List of 'big' **degenerate** sets that are certainly *not* contained
    # C_degBig = --- ; % capsules can not be degenerate (commented out in MATLAB)
    I_degBig = Interval(np.array([-100, 0]), np.array([100, 0]))
    cPZ_degBig = ConPolyZono(I_degBig)
    cZ_degBig = ConZonotope(I_degBig)
    # E_degBig = [1 0; 0 0] * ellipsoid(100*eye(2))
    # MATLAB: E_degBig = [1 0; 0 0] * ellipsoid(100*eye(2))
    # Python: Use @ operator (matrix multiplication) - Ellipsoid should support __rmatmul__
    E_degBig = np.array([[1, 0], [0, 0]]) @ Ellipsoid(100 * np.eye(2))
    # LevelSet: MATLAB uses syms x y; eqs = [x^2 + y^2 - 100; y; -y]; ls_degBig = levelSet(eqs,[x;y],'<=')
    # ls_degBig = LevelSet(...)  # TODO: Implement symbolic level set creation
    P_degBig = 100 * Polytope(np.array([[-1, 1], [0, 0]]))
    pZ_degBig = PolyZonotope(I_degBig)
    SpS_degBig = SpectraShadow(I_degBig)
    zB_degBig = ZonoBundle(I_degBig)
    Z_degBig = Zonotope(I_degBig)
    
    degBigSets = [I_degBig, cPZ_degBig, cZ_degBig, E_degBig,
                  P_degBig, pZ_degBig, SpS_degBig, zB_degBig, Z_degBig]
    # Note: C_degBig (Capsule) is not included as capsules cannot be degenerate
    # Note: ls_degBig (LevelSet) is commented out as it requires symbolic math
    
    # List of empty instances of all sets
    C_empty = Capsule.empty(2)
    I_empty = Interval.empty(2)
    cPZ_empty = ConPolyZono.empty(2)
    cZ_empty = ConZonotope.empty(2)
    E_empty = Ellipsoid.empty(2)
    ls_empty = LevelSet.empty(2)
    P_empty = Polytope.empty(2)
    pZ_empty = PolyZonotope.empty(2)
    SpS_empty = SpectraShadow.empty(2)
    zB_empty = ZonoBundle.empty(2)
    Z_empty = Zonotope.empty(2)
    
    emptySets = [C_empty, I_empty, cPZ_empty, cZ_empty, E_empty, ls_empty,
                P_empty, pZ_empty, SpS_empty, zB_empty, Z_empty]
    # Filter out None values (sets that don't support empty instantiation)
    emptySets = [s for s in emptySets if s is not None]
    
    # We can now perform all the actual containment checks
    
    # Empty set and fullspace containment
    aux_executeContainmentChecks_emptySet(S, Sdeg, Sempty, additionalAlgorithms)
    aux_executeContainmentChecks_fullspace(S, Sdeg, Sempty, additionalAlgorithms)
    
    # Point-containment
    exactIsImplemented = False
    try:
        # Origin must be contained, unless S is empty
        res = S.contains_(point_contained, 'exact', tol)[0]
        res_cert, cert = S.contains_(point_contained, 'exact', tol)[:2]
        
        if hasattr(S, 'representsa_') and S.representsa_('emptySet'):
            assert not res
            assert not res_cert
            assert cert
        else:
            assert res
            assert res_cert
            assert cert
        
        # Far away point must not be contained, except if S is the fullspace
        res = S.contains_(point_notContained, 'exact', tol)[0]
        res_cert, cert = S.contains_(point_notContained, 'exact', tol)[:2]
        
        if hasattr(S, 'representsa_') and S.representsa_('fullspace'):
            assert res
            assert res_cert
        else:
            assert not res
            assert not res_cert
        assert cert
        
        exactIsImplemented = True
    except Exception as ME:
        # The code above is only allowed to fail if there is no exact algorithm for point-containment
        if 'point' not in setsNonExact:
            raise ME
    
    # If the point-containment behaves like it is exact, make sure that this is intended
    assert not (exactIsImplemented and 'point' in setsNonExact)
    
    # There **MUST** be at least an approximative way of checking for point-containment
    S.contains_(point_contained, 'approx')
    S.contains_(point_contained, 'approx')[:2]
    
    # Basic containment checks
    # If S is non-degenerate, it must contain smallSets, otherwise not
    shouldBeContained = S.isFullDim(tol) if hasattr(S, 'isFullDim') else True
    aux_executeContainmentChecks_basic(S, smallSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    
    # S must contain degSmallSets, emptySets
    shouldBeContained = True
    aux_executeContainmentChecks_basic(S, degSmallSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    aux_executeContainmentChecks_basic(S, emptySets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    
    # S must *not* contain bigSets, degBigSets
    shouldBeContained = False
    aux_executeContainmentChecks_basic(S, bigSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    aux_executeContainmentChecks_basic(S, degBigSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    
    # Sdeg must contain emptySets
    shouldBeContained = True
    aux_executeContainmentChecks_basic(Sdeg, emptySets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    
    # If Sdeg is not empty, it must contain smallDegSets, otherwise not
    shouldBeContained = not (hasattr(Sdeg, 'representsa_') and Sdeg.representsa_('emptySet'))
    aux_executeContainmentChecks_basic(Sdeg, degSmallSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    
    # Sdeg must *not* contain smallSets, bigSets, degBigSets
    shouldBeContained = False
    aux_executeContainmentChecks_basic(Sdeg, smallSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    aux_executeContainmentChecks_basic(Sdeg, bigSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    aux_executeContainmentChecks_basic(Sdeg, degBigSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    
    # Sempty must contain emptySets
    shouldBeContained = True
    aux_executeContainmentChecks_basic(Sempty, emptySets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    
    # Sempty must *not* contain smallSets, smallDegSets, bigSets, degBigSets
    shouldBeContained = False
    aux_executeContainmentChecks_basic(Sempty, smallSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    aux_executeContainmentChecks_basic(Sempty, degSmallSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    aux_executeContainmentChecks_basic(Sempty, bigSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)
    aux_executeContainmentChecks_basic(Sempty, degBigSets, shouldBeContained, implementedSets, 
                                      setsNonExact, additionalAlgorithms, 
                                      additionalAlgorithms_specificSets)


def aux_executeContainmentChecks_emptySet(S, Sdeg, Sempty, additionalAlgorithms):
    """Checks if S, Sdeg, Sempty contain the empty set, and whether S and Sdeg are
    *not* contained in the empty set, but Sempty is contained in the empty set."""
    from cora_python.contSet.emptySet.emptySet import EmptySet
    
    es = EmptySet(2)
    
    algorithms = ['exact', 'approx'] + (additionalAlgorithms if additionalAlgorithms else [])
    
    for algo in algorithms:
        # Randomly disable test (10% chance to run)
        if random.random() > 0.1:
            continue
        
        # No matter the algorithm, the empty set needs to be contained
        S_res = S.contains_(es, algo)[0]
        S_res_cert, S_cert = S.contains_(es, algo)[:2]
        Sdeg_res = Sdeg.contains_(es, algo)[0]
        # MATLAB line 327 has: [Sdeg_res_cert, Sdeg_cert] = contains(S,es,algo);
        # This appears to be a bug (should be Sdeg), but we match MATLAB exactly
        Sdeg_res_cert, Sdeg_cert = S.contains_(es, algo)[:2]
        Sempty_res = Sempty.contains_(es, algo)[0]
        Sempty_res_cert, Sempty_cert = Sempty.contains_(es, algo)[:2]
        
        res = S_res and S_res_cert and Sdeg_res and Sdeg_res_cert and Sempty_res and Sempty_res_cert
        cert = S_cert and Sdeg_cert and Sempty_cert
        
        assert res
        assert cert
    
    # Randomly disable test
    if random.random() > 0.1:
        return
    
    # Now, we check that S is not contained in the empty set, but that Sempty is contained
    S_res = es.contains_(S)[0]
    S_res_cert, S_cert = es.contains_(S)[:2]
    S_res_scaling, S_cert_scaling, S_scaling = es.contains_(S)
    
    Sdeg_res = es.contains_(Sdeg)[0]
    Sdeg_res_cert, Sdeg_cert = es.contains_(Sdeg)[:2]
    Sdeg_res_scaling, Sdeg_cert_scaling, Sdeg_scaling = es.contains_(Sdeg)
    
    Sempty_res = es.contains_(Sempty)[0]
    Sempty_res_cert, Sempty_cert = es.contains_(Sempty)[:2]
    Sempty_res_scaling, Sempty_cert_scaling, Sempty_scaling = es.contains_(Sempty)
    
    degIsEmpty = hasattr(Sdeg, 'representsa_') and Sdeg.representsa_('emptySet')
    
    # If S is empty itself, we can skip the first check
    if hasattr(S, 'representsa_') and S.representsa_('emptySet'):
        res = True
    else:
        res = not S_res and not S_res_cert and not S_res_scaling
    
    res = res and (Sdeg_res == degIsEmpty) and (Sdeg_res_cert == degIsEmpty) and (Sdeg_res_scaling == degIsEmpty)
    res = res and Sempty_res and Sempty_res_cert and Sempty_res_scaling
    
    cert = S_cert and S_cert_scaling and Sdeg_cert and Sdeg_cert_scaling and Sempty_cert and Sempty_cert_scaling
    
    # If S is the empty set, we need a special check
    if hasattr(S, 'representsa_') and S.representsa_('emptySet'):
        scaling = (S_scaling == 0)
    else:
        scaling = (S_scaling == np.inf)  # S cannot be contained in any scaling of the empty set
    
    if degIsEmpty:
        scaling = scaling and (Sdeg_scaling == 0)
    else:
        scaling = scaling and (Sdeg_scaling == np.inf)
    
    scaling = scaling and (Sempty_scaling == 0)
    
    assert res
    assert cert
    assert scaling


def aux_executeContainmentChecks_fullspace(S, Sdeg, Sempty, additionalAlgorithms):
    """Checks if S, Sdeg, Sempty do *not* contain the full space, and whether
    S, Sdeg, Sempty are contained in the fullspace."""
    from cora_python.contSet.fullspace.fullspace import Fullspace
    
    tol = 1e-6
    fs = Fullspace(2)
    
    algorithms = ['exact', 'approx'] + (additionalAlgorithms if additionalAlgorithms else [])
    
    for algo in algorithms:
        # Randomly disable test
        if random.random() > 0.1:
            continue
        
        # No matter the algorithm, the full space can *not* be contained (except if it is the fullspace)
        S_res = S.contains_(fs, algo)[0]
        S_res_cert, S_cert = S.contains_(fs, algo)[:2]
        Sdeg_res = Sdeg.contains_(fs, algo)[0]
        Sdeg_res_cert, Sdeg_cert = Sdeg.contains_(fs, algo)[:2]
        Sempty_res = Sempty.contains_(fs, algo)[0]
        Sempty_res_cert, Sempty_cert = Sempty.contains_(fs, algo)[:2]
        
        if hasattr(S, 'representsa_') and S.representsa_('fullspace'):
            res = S_res and S_res_cert and not Sdeg_res and not Sdeg_res_cert and not Sempty_res and not Sempty_res_cert
        else:
            res = not S_res and not S_res_cert and not Sdeg_res and not Sdeg_res_cert and not Sempty_res and not Sempty_res_cert
        
        cert = S_cert and Sdeg_cert and Sempty_cert
        
        assert res
        assert cert
    
    # Randomly disable test
    if random.random() > 0.1:
        return
    
    # Now, we check that S, Sdeg, Sempty are contained in the full space
    S_res = fs.contains_(S)[0]
    S_res_cert, S_cert = fs.contains_(S)[:2]
    S_res_scaling, S_cert_scaling, S_scaling = fs.contains_(S)
    
    Sdeg_res = fs.contains_(Sdeg)[0]
    Sdeg_res_cert, Sdeg_cert = fs.contains_(Sdeg)[:2]
    Sdeg_res_scaling, Sdeg_cert_scaling, Sdeg_scaling = fs.contains_(Sdeg)
    
    Sempty_res = fs.contains_(Sempty)[0]
    Sempty_res_cert, Sempty_cert = fs.contains_(Sempty)[:2]
    Sempty_res_scaling, Sempty_cert_scaling, Sempty_scaling = fs.contains_(Sempty)
    
    res = S_res and S_res_cert and S_res_scaling
    res = res and Sdeg_res and Sdeg_res_cert and Sdeg_res_scaling
    res = res and Sempty_res and Sempty_res_cert and Sempty_res_scaling
    
    cert = S_cert and S_cert_scaling and Sdeg_cert and Sdeg_cert_scaling and Sempty_cert and Sempty_cert_scaling
    
    scaling = (S_scaling == 0)  # S is contained in any scaling of fullspace, so the infimum is 0
    scaling = scaling and (Sdeg_scaling == 0)
    scaling = scaling and (Sempty_scaling == 0)
    
    assert res
    assert cert
    assert scaling


def aux_executeContainmentChecks_basic(basicSet, setCollection, shouldBeContained, 
                                       implementedSets, setsNonExact, additionalAlgorithms, 
                                       additionalAlgorithms_specificSets):
    """Checks if basicSet contains (or not, depending on shouldBeContained) all sets
    from setCollection, if they are implemented."""
    tol = 1e-6
    N = 3  # Very low number of iterations for opt and sampling methods
    
    for set_obj in setCollection:
        # Randomly disable test
        if random.random() > 0.1:
            continue
        
        exactIsImplemented = False
        # Get class name before try block (needed for exception handling)
        set_class_name = set_obj.__class__.__name__
        
        try:
            res = basicSet.contains_(set_obj, 'exact', tol)[0]
            res_cert, cert = basicSet.contains_(set_obj, 'exact', tol)[:2]
            
            if hasattr(set_obj, 'representsa_') and set_obj.representsa_('emptySet'):
                # Empty sets must always be contained
                assert res == True
                assert res_cert == True
                assert cert
            elif hasattr(basicSet, 'representsa_') and basicSet.representsa_('emptySet'):
                # The empty set contains nothing, except the empty set
                assert res == False
                assert res_cert == False
                assert cert
            elif hasattr(basicSet, 'representsa_') and basicSet.representsa_('fullspace'):
                # The fullspace contains everything
                assert res == True
                assert res_cert == True
                assert cert
            else:
                assert res == shouldBeContained
                assert res_cert == shouldBeContained
                assert cert
            
            # Also, check if the 'approx' method works
            basicSet.contains_(set_obj, 'approx')
            basicSet.contains_(set_obj, 'approx')[:2]
            
            exactIsImplemented = True
        
        except Exception as ME:
            # The containment check may only fail if:
            if set_class_name not in implementedSets:
                # 1) the containment check is not implemented at all -> ignore
                continue
            elif set_class_name in setsNonExact:
                # 2) the exact containment check is not available -> make sure that at least the 'approx' algorithm works
                basicSet.contains_(set_obj, 'approx')
                basicSet.contains_(set_obj, 'approx')[:2]
            else:
                raise ME
        
        # Also need to verify that the set is non-empty
        isEmpty = (hasattr(set_obj, 'representsa_') and set_obj.representsa_('emptySet')) or \
                  (hasattr(basicSet, 'representsa_') and basicSet.representsa_('emptySet'))
        
        # If the containment behaves like it is exact, make sure that this is intended
        if not isEmpty:
            assert not (exactIsImplemented and set_class_name in setsNonExact)
        else:
            # Empty sets must always be recognized
            assert exactIsImplemented
    
    # It only remains to check the additional algorithms
    if additionalAlgorithms:
        for i, algorithm in enumerate(additionalAlgorithms):
            # Randomly disable test
            if random.random() > 0.1:
                continue
            
            algorithm_implementedSets = additionalAlgorithms_specificSets[i] if i < len(additionalAlgorithms_specificSets) else []
            
            for set_obj in setCollection:
                set_class_name = set_obj.__class__.__name__
                
                algorithmIsImplemented = False
                try:
                    res = basicSet.contains_(set_obj, algorithm, tol, N)[0]
                    res_cert, cert = basicSet.contains_(set_obj, algorithm, tol, N)[:2]
                    
                    algorithmIsImplemented = True
                
                except Exception as ME:
                    # The containment check may only fail if it is not implemented
                    if algorithm_implementedSets and set_class_name not in algorithm_implementedSets:
                        continue
                    else:
                        raise ME
                
                # Also need to verify that the set is non-empty
                isEmpty = (hasattr(set_obj, 'representsa_') and set_obj.representsa_('emptySet')) or \
                          (hasattr(basicSet, 'representsa_') and basicSet.representsa_('emptySet'))
                
                # If the containment behaves like it is implemented, make sure that this is intended
                # MATLAB: checkToFail = algorithmIsImplemented && any(strcmp(algorithm_implementedSets, class(set))) && isempty(algorithm_implementedSets);
                # This checks: if algorithm is implemented AND set class is in algorithm_implementedSets AND algorithm_implementedSets is empty
                # Note: The logic seems odd (empty list can't contain anything), but we match MATLAB exactly
                checkToFail = algorithmIsImplemented and (set_class_name in algorithm_implementedSets if algorithm_implementedSets else False) and (not algorithm_implementedSets)
                if not isEmpty:
                    assert not checkToFail
                else:
                    # Empty sets must always be recognized
                    assert algorithmIsImplemented

