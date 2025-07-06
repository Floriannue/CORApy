import pytest
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.specification.stlInterval import StlInterval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestIntervalStlInterval:
    """Test class for interval stlInterval method"""
    
    def test_stlInterval_1d_interval(self):
        """Test stlInterval conversion with 1D interval"""
        # Create 1D interval
        I = Interval([1], [2])
        
        # Convert to stlInterval
        result = I.stlInterval()
        
        # Check result
        assert isinstance(result, StlInterval)
        assert result.lower == 1
        assert result.upper == 2
        assert result.leftClosed == True
        assert result.rightClosed == True
    
    def test_stlInterval_point_interval(self):
        """Test stlInterval conversion with point interval"""
        # Create point interval
        I = Interval([1.5], [1.5])
        
        # Convert to stlInterval
        result = I.stlInterval()
        
        # Check result
        assert isinstance(result, StlInterval)
        assert result.lower == 1.5
        assert result.upper == 1.5
        assert result.leftClosed == True
        assert result.rightClosed == True
    
    def test_stlInterval_multidimensional_error(self):
        """Test that multidimensional intervals throw error"""
        # Create 2D interval
        I = Interval([1, 2], [3, 4])
        
        # Should throw error
        with pytest.raises(CORAerror) as exc_info:
            I.stlInterval()
        
        assert 'one-dimensional' in str(exc_info.value)
    
    def test_stlInterval_zero_width_interval(self):
        """Test stlInterval with zero-width interval"""
        # Create zero-width interval
        I = Interval([0], [0])
        
        # Convert to stlInterval
        result = I.stlInterval()
        
        # Check result
        assert isinstance(result, StlInterval)
        assert result.lower == 0
        assert result.upper == 0
        assert result.leftClosed == True
        assert result.rightClosed == True
    
    def test_stlInterval_large_interval(self):
        """Test stlInterval with large interval"""
        # Create large interval
        I = Interval([0], [1000])
        
        # Convert to stlInterval
        result = I.stlInterval()
        
        # Check result
        assert isinstance(result, StlInterval)
        assert result.lower == 0
        assert result.upper == 1000
        assert result.leftClosed == True
        assert result.rightClosed == True


def test_interval_stlInterval():
    """Basic test for interval stlInterval method"""
    # Create 1D interval
    I = Interval([2], [5])
    
    # Convert to stlInterval
    result = I.stlInterval()
    
    # Verify result
    assert isinstance(result, StlInterval)
    assert result.lower == 2
    assert result.upper == 5
    assert result.leftClosed == True
    assert result.rightClosed == True 