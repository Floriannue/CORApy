"""
Test cases for projectHighDim_ method of EmptySet class
"""

import pytest
from cora_python.contSet.emptySet import EmptySet


def test_projectHighDim_basic():
    """Test basic projectHighDim functionality"""
    O = EmptySet(2)
    original_dim = O.dimension
    
    # Lift to higher dimension
    O_high = O.projectHighDim_(5, [1, 2])
    
    # Should have updated dimension
    assert O_high.dimension == 5
    # Should be the same object (modifies in place)
    assert O_high is O


def test_projectHighDim_same_dimension():
    """Test projectHighDim to same dimension"""
    O = EmptySet(3)
    
    O_same = O.projectHighDim_(3, [1, 2, 3])
    
    assert O_same.dimension == 3
    assert O_same is O


def test_projectHighDim_different_projections():
    """Test projectHighDim with different projection mappings"""
    O = EmptySet(2)
    
    # Test with different projection mappings
    O_high1 = O.projectHighDim_(4, [1, 3])
    assert O_high1.dimension == 4
    
    O_high2 = O.projectHighDim_(6, [2, 5])
    assert O_high2.dimension == 6


def test_projectHighDim_large_dimension():
    """Test projectHighDim to large dimension"""
    O = EmptySet(1)
    
    O_large = O.projectHighDim_(100, [50])
    
    assert O_large.dimension == 100
    assert O_large is O


def test_projectHighDim_return_value():
    """Test that projectHighDim returns the object"""
    O = EmptySet(2)
    
    result = O.projectHighDim_(5, [1, 2])
    
    # Should return the same object
    assert result is O
    assert result.dimension == 5


def test_projectHighDim_multiple_calls():
    """Test multiple calls to projectHighDim"""
    O = EmptySet(2)
    
    # First call
    O1 = O.projectHighDim_(4, [1, 2])
    assert O1.dimension == 4
    
    # Second call
    O2 = O.projectHighDim_(6, [1, 2])
    assert O2.dimension == 6
    
    # Should be the same object
    assert O1 is O2 is O 