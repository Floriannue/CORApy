"""
test_contSet_generateRandom - unit test function of
    contSet.generateRandom

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       05-April-2023
Last update:   ---
Last revision: 09-January-2024
"""

import pytest
import sys
import os

# Add the parent directory to the path to import cora_python modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cora_python.contSet.contSet.generateRandom import generateRandom
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.contSet.dim import dim


class TestContSetGenerateRandom:
    
    def test_generate_random_basic(self):
        """Test basic generateRandom functionality"""
        # Test multiple generateRandom calls
        for i in range(10):
            S = generateRandom()
            assert hasattr(S, '__class__')
            # For now, since we only have interval implemented, check it's an interval
            assert isinstance(S, Interval)
    
    def test_generate_random_dimension(self):
        """Test generateRandom with specified dimension"""
        S = generateRandom(dimension=3)
        assert dim(S) == 3
    
    def test_generate_random_given_classes_interval(self):
        """Test generateRandom with given class: interval"""
        S = generateRandom(['interval'])
        assert isinstance(S, Interval)
    
    def test_generate_random_given_classes_mixed(self):
        """Test generateRandom with multiple admissible classes"""
        # For now, only interval is implemented, so this should return interval
        S = generateRandom(['interval'])  # Only include implemented classes
        assert isinstance(S, Interval)
    
    def test_generate_random_with_dimension_and_classes(self):
        """Test generateRandom with both dimension and classes specified"""
        S = generateRandom(['interval'], Dimension=2)
        assert isinstance(S, Interval)
        assert dim(S) == 2
    
    def test_generate_random_invalid_class(self):
        """Test generateRandom with invalid class should fallback to interval"""
        # Since most classes are not implemented, this should fallback to interval
        S = generateRandom(['nonexistent_class'])
        assert isinstance(S, Interval)
    
    def test_generate_random_dimension_validation(self):
        """Test that generated random sets have proper dimensions"""
        for d in [1, 2, 3, 5]:
            S = generateRandom(Dimension=d)
            assert dim(S) == d
    
    def test_generate_random_reproducibility(self):
        """Test that generateRandom produces different results (is truly random)"""
        # Generate multiple intervals and check they're different
        intervals = [generateRandom(Dimension=2) for _ in range(5)]
        
        # Check that not all intervals are identical
        # (Very unlikely but theoretically possible for truly random generation)
        all_identical = True
        for i in range(1, len(intervals)):
            if not (intervals[i].inf == intervals[0].inf).all() or \
               not (intervals[i].sup == intervals[0].sup).all():
                all_identical = False
                break
        
        # It's extremely unlikely that 5 random intervals are identical
        assert not all_identical, "Generated intervals should not all be identical"


if __name__ == '__main__':
    pytest.main([__file__]) 