"""
test_verifyTime - unit tests for VerifyTime class

This module contains comprehensive unit tests for the VerifyTime class
used for time interval management in specifications.

Authors: Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np
from cora_python.g.classes.verifyTime import VerifyTime
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestVerifyTime:
    """Test class for VerifyTime"""
    
    def test_init_empty(self):
        """Test initialization with no arguments"""
        vt = VerifyTime()
        assert hasattr(vt, 'intervals')
        assert isinstance(vt.intervals, list)
        assert len(vt.intervals) == 0
    
    def test_init_single_interval(self):
        """Test initialization with single interval"""
        vt = VerifyTime([0, 5])
        assert len(vt.intervals) == 1
        assert vt.intervals[0] == [0, 5]
    
    def test_init_multiple_intervals(self):
        """Test initialization with multiple intervals"""
        intervals = [[0, 2], [4, 6], [8, 10]]
        vt = VerifyTime(intervals)
        assert len(vt.intervals) == 3
        assert vt.intervals == intervals
    
    def test_init_numpy_array(self):
        """Test initialization with numpy array"""
        intervals = np.array([[0, 2], [4, 6]])
        vt = VerifyTime(intervals)
        assert len(vt.intervals) == 2
        # Should convert to list format
        assert isinstance(vt.intervals, list)
    
    def test_numIntervals(self):
        """Test numIntervals method"""
        # Empty
        vt = VerifyTime()
        assert vt.numIntervals() == 0
        
        # Single interval
        vt = VerifyTime([0, 5])
        assert vt.numIntervals() == 1
        
        # Multiple intervals
        vt = VerifyTime([[0, 2], [4, 6], [8, 10]])
        assert vt.numIntervals() == 3
    
    def test_timeUntilSwitch_empty(self):
        """Test timeUntilSwitch with empty intervals"""
        vt = VerifyTime()
        max_time, full_comp = vt.timeUntilSwitch(1.0)
        assert max_time == float('inf')
        assert full_comp == False
    
    def test_timeUntilSwitch_single_interval(self):
        """Test timeUntilSwitch with single interval"""
        vt = VerifyTime([2, 8])
        
        # Before interval
        max_time, full_comp = vt.timeUntilSwitch(1.0)
        assert max_time == 1.0  # 2 - 1
        assert full_comp == False
        
        # Inside interval
        max_time, full_comp = vt.timeUntilSwitch(5.0)
        assert max_time == 3.0  # 8 - 5
        assert full_comp == True
        
        # After interval
        max_time, full_comp = vt.timeUntilSwitch(10.0)
        assert max_time == float('inf')
        assert full_comp == False
    
    def test_timeUntilSwitch_multiple_intervals(self):
        """Test timeUntilSwitch with multiple intervals"""
        vt = VerifyTime([[1, 3], [5, 7], [9, 11]])
        
        # Before first interval
        max_time, full_comp = vt.timeUntilSwitch(0.5)
        assert max_time == 0.5  # 1 - 0.5
        assert full_comp == False
        
        # Inside first interval
        max_time, full_comp = vt.timeUntilSwitch(2.0)
        assert max_time == 1.0  # 3 - 2
        assert full_comp == True
        
        # Between intervals
        max_time, full_comp = vt.timeUntilSwitch(4.0)
        assert max_time == 1.0  # 5 - 4
        assert full_comp == False
        
        # Inside second interval
        max_time, full_comp = vt.timeUntilSwitch(6.0)
        assert max_time == 1.0  # 7 - 6
        assert full_comp == True
    
    def test_contains_time(self):
        """Test if time is contained in intervals"""
        vt = VerifyTime([[1, 3], [5, 7]])
        
        # Test helper method if it exists
        if hasattr(vt, 'contains'):
            assert vt.contains(2.0) == True
            assert vt.contains(6.0) == True
            assert vt.contains(0.5) == False
            assert vt.contains(4.0) == False
            assert vt.contains(8.0) == False
    
    def test_merge_overlapping_intervals(self):
        """Test handling of overlapping intervals"""
        
        # Overlapping intervals should raise an error (matching MATLAB behavior)
        with pytest.raises(CORAerror):
            vt = VerifyTime([[1, 3], [2, 5], [7, 9]])
        
        # Non-overlapping intervals should work fine
        vt = VerifyTime([[1, 3], [4, 5], [7, 9]])
        assert vt.numIntervals() == 3
        
        # If merge functionality exists, test it with adjacent intervals
        if hasattr(vt, 'compact'):
            # Adjacent intervals that can be compacted
            vt_adjacent = VerifyTime([[1, 3], [3, 5], [7, 9]])
            compacted = vt_adjacent.compact()
            # Should merge [1,3] and [3,5] into [1,5]
            assert compacted.numIntervals() == 2
    
    def test_intersection_with_interval(self):
        """Test intersection with another interval if implemented"""
        vt = VerifyTime([[1, 3], [5, 7]])
        
        # Test intersection functionality if it exists
        if hasattr(vt, 'intersect'):
            result = vt.intersect([2, 6])
            # Should return [[2, 3], [5, 6]]
            pass  # Implementation would depend on actual method
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Zero-length interval
        vt = VerifyTime([2, 2])
        assert vt.numIntervals() == 1
        
        # Very small interval
        vt = VerifyTime([1.0, 1.0001])
        assert vt.numIntervals() == 1
        
        # Large time values
        vt = VerifyTime([1e6, 1e7])
        max_time, full_comp = vt.timeUntilSwitch(5e5)
        assert max_time == 5e5  # 1e6 - 5e5
        assert full_comp == False
    
    def test_invalid_intervals(self):
        """Test handling of invalid intervals"""
        
        # Interval with start > end - should raise CORAerror
        with pytest.raises(CORAerror):
            vt = VerifyTime([5, 2])  # Invalid: start > end
        
        # Empty interval list - should work
        vt = VerifyTime([])
        assert vt.numIntervals() == 0
    
    def test_str_representation(self):
        """Test string representation if implemented"""
        vt = VerifyTime([[1, 3], [5, 7]])
        
        if hasattr(vt, '__str__'):
            str_repr = str(vt)
            assert isinstance(str_repr, str)
            assert len(str_repr) > 0
        
        if hasattr(vt, '__repr__'):
            repr_str = repr(vt)
            assert isinstance(repr_str, str)
            assert len(repr_str) > 0
    
    def test_equality(self):
        """Test equality comparison if implemented"""
        vt1 = VerifyTime([[1, 3], [5, 7]])
        vt2 = VerifyTime([[1, 3], [5, 7]])
        vt3 = VerifyTime([[1, 3], [5, 8]])
        
        if hasattr(vt1, '__eq__'):
            assert vt1 == vt2
            assert vt1 != vt3
    
    def test_copy(self):
        """Test copying functionality if implemented"""
        vt = VerifyTime([[1, 3], [5, 7]])
        
        if hasattr(vt, 'copy'):
            vt_copy = vt.copy()
            assert vt_copy.intervals == vt.intervals
            # Modify original and ensure copy is unchanged
            vt.intervals.append([9, 11])
            assert len(vt_copy.intervals) == 2
            assert len(vt.intervals) == 3


def test_verifyTime_integration():
    """Integration test for VerifyTime with typical usage patterns"""
    # Create time intervals for specification checking
    vt = VerifyTime([[0, 2], [4, 6], [8, 10]])
    
    # Simulate time progression
    times = [0.5, 1.5, 3.0, 4.5, 7.0, 8.5, 12.0]
    
    for t in times:
        max_time, full_comp = vt.timeUntilSwitch(t)
        
        # Basic checks
        assert isinstance(max_time, (int, float))
        assert isinstance(full_comp, bool)
        assert max_time >= 0 or max_time == float('inf')
        
        # If inside an interval, should require full computation
        inside_interval = any(start <= t <= end for start, end in vt.intervals)
        if inside_interval:
            assert full_comp == True
        
        print(f"Time {t}: max_time={max_time}, full_comp={full_comp}")


if __name__ == '__main__':
    test = TestVerifyTime()
    test.test_init_empty()
    test.test_init_single_interval()
    test.test_init_multiple_intervals()
    test.test_numIntervals()
    test.test_timeUntilSwitch_empty()
    test.test_timeUntilSwitch_single_interval()
    test.test_timeUntilSwitch_multiple_intervals()
    test.test_edge_cases()
    test.test_invalid_intervals()
    
    # Run integration test
    test_verifyTime_integration()
    
    print("All VerifyTime tests passed!") 