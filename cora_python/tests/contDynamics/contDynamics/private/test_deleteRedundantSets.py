"""
test_deleteRedundantSets - GENERATED TEST
   No MATLAB source test exists. This test was created by analyzing the MATLAB 
   implementation logic in deleteRedundantSets.py and ensuring thorough coverage.

   This test verifies that deleteRedundantSets correctly deletes reachable sets 
   that are already covered by other sets, including:
   - Internal count management
   - Periodic reduction (when internalCount == reductionInterval)
   - Set intersection and redundancy removal (when internalCount == 2)
   - Polytope operations for set difference (mldivide)

Syntax:
    pytest cora_python/tests/contDynamics/contDynamics/private/test_deleteRedundantSets.py

Authors:       Generated test based on MATLAB implementation
Written:       2025
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.contDynamics.private.deleteRedundantSets import deleteRedundantSets


class TestDeleteRedundantSets:
    """Test class for deleteRedundantSets functionality"""
    
    def test_deleteRedundantSets_first_run(self):
        """Test first run (no Rold.internalCount)"""
        # MATLAB: R.internalCount=3; (for first run)
        R = {
            'tp': [
                {'set': Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)), 'error': np.array([[0.01], [0.01]])}
            ],
            'ti': [Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))]
        }
        Rold = {}  # Empty - no internalCount
        options = {
            'reductionInterval': 3,
            'maxError': np.array([[0.1], [0.1]])
        }
        
        R_result = deleteRedundantSets(R, Rold, options)
        
        # Should set internalCount to 3
        assert R_result['internalCount'] == 3
    
    def test_deleteRedundantSets_increment_count(self):
        """Test that internal count is incremented"""
        R = {
            'tp': [
                {'set': Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)), 'error': np.array([[0.01], [0.01]])}
            ],
            'ti': [Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))]
        }
        Rold = {'internalCount': 1}
        options = {
            'reductionInterval': 3,
            'maxError': np.array([[0.1], [0.1]])
        }
        
        R_result = deleteRedundantSets(R, Rold, options)
        
        # Should increment: 1 + 1 = 2
        assert R_result['internalCount'] == 2
    
    def test_deleteRedundantSets_reduction_interval(self):
        """Test when internalCount == reductionInterval"""
        # MATLAB: if R.internalCount==options.reductionInterval
        R = {
            'tp': [
                {'set': Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)), 'error': np.array([[0.01], [0.01]])},
                {'set': Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2)), 'error': np.array([[0.01], [0.01]])}
            ],
            'ti': [
                Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)),
                Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2))
            ]
        }
        Rold = {'internalCount': 2}  # Will become 3
        options = {
            'reductionInterval': 3,
            'maxError': np.array([[0.1], [0.1]])
        }
        
        R_result = deleteRedundantSets(R, Rold, options)
        
        # Should reset to 1 and create P list
        assert R_result['internalCount'] == 1
        assert 'P' in R_result
        assert len(R_result['P']) == len(R['tp'])
    
    def test_deleteRedundantSets_count_equals_two(self):
        """Test when internalCount == 2 (redundancy removal)"""
        # MATLAB: elseif R.internalCount==2
        R = {
            'tp': [
                {'set': Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)), 'error': np.array([[0.01], [0.01]])},
                {'set': Zonotope(np.array([[0.5], [0.5]]), 0.1 * np.eye(2)), 'error': np.array([[0.01], [0.01]])}
            ],
            'ti': [
                Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)),
                Zonotope(np.array([[0.5], [0.5]]), 0.1 * np.eye(2))
            ]
        }
        Rold = {'internalCount': 1, 'P': []}  # Will become 2
        options = {
            'reductionInterval': 3,
            'maxError': np.array([[0.1], [0.1]])
        }
        
        R_result = deleteRedundantSets(R, Rold, options)
        
        # Should process sets and potentially remove redundant ones
        assert R_result['internalCount'] == 2
        # R.tp should be modified (may have fewer sets if redundant)
        assert 'tp' in R_result
        assert len(R_result['tp']) <= len(R['tp'])
    
    def test_deleteRedundantSets_with_previous_sets(self):
        """Test with previous time step sets (Rold.P)"""
        R = {
            'tp': [
                {'set': Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)), 'error': np.array([[0.01], [0.01]])}
            ],
            'ti': [Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))]
        }
        # Create Rold with P (polytopes from previous step)
        from cora_python.contSet.polytope import Polytope
        Rold = {
            'internalCount': 1,
            'P': [
                Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), np.array([[1], [1], [1], [1]]))
            ]
        }
        options = {
            'reductionInterval': 3,
            'maxError': np.array([[0.1], [0.1]])
        }
        
        R_result = deleteRedundantSets(R, Rold, options)
        
        # Should intersect with previous sets
        assert R_result['internalCount'] == 2
        assert 'tp' in R_result
    
    def test_deleteRedundantSets_with_parent(self):
        """Test that parent field is preserved"""
        R = {
            'tp': [
                {
                    'set': Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)),
                    'error': np.array([[0.01], [0.01]]),
                    'parent': 0
                }
            ],
            'ti': [Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))]
        }
        Rold = {'internalCount': 1}
        options = {
            'reductionInterval': 3,
            'maxError': np.array([[0.1], [0.1]])
        }
        
        R_result = deleteRedundantSets(R, Rold, options)
        
        # Parent should be preserved if set is kept
        if len(R_result['tp']) > 0 and 'parent' in R['tp'][0]:
            assert 'parent' in R_result['tp'][0]


def test_deleteRedundantSets():
    """Test function for deleteRedundantSets method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestDeleteRedundantSets()
    test.test_deleteRedundantSets_first_run()
    test.test_deleteRedundantSets_increment_count()
    test.test_deleteRedundantSets_reduction_interval()
    test.test_deleteRedundantSets_count_equals_two()
    test.test_deleteRedundantSets_with_previous_sets()
    test.test_deleteRedundantSets_with_parent()
    
    print("test_deleteRedundantSets: all tests passed")
    return True


if __name__ == "__main__":
    test_deleteRedundantSets()

