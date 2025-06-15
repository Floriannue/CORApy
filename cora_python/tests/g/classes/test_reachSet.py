"""
test_reachSet - unit test function for reachSet class

Tests the reachSet class for storing reachable sets.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.classes.reachSet import ReachSet
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestReachSet:
    def test_reachSet_constructor_empty(self):
        """Test empty constructor"""
        R = ReachSet()
        assert R.timePoint == {}
        assert R.timeInterval == {}
    
    def test_reachSet_constructor_timePoint_only(self):
        """Test constructor with only timePoint"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2))],
            'time': [0.0]
        }
        R = ReachSet(timePoint)
        assert R.timePoint == timePoint
        assert R.timeInterval == {}
    
    def test_reachSet_constructor_both(self):
        """Test constructor with both timePoint and timeInterval"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2))],
            'time': [0.0]
        }
        timeInterval = {
            'set': [Zonotope([1, 2], np.eye(2) * 0.1)],
            'time': [[0.0, 0.1]]
        }
        R = ReachSet(timePoint, timeInterval)
        assert R.timePoint == timePoint
        assert R.timeInterval == timeInterval
    
    def test_initReachSet_static_method(self):
        """Test static initReachSet method"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2))],
            'time': [0.0]
        }
        R = ReachSet.initReachSet(timePoint)
        assert isinstance(R, ReachSet)
        assert R.timePoint == timePoint
        assert R.timeInterval == {}
    
    def test_query_reachSet(self):
        """Test query method for reachSet"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2))],
            'time': [0.0]
        }
        R = ReachSet(timePoint)
        
        result = R.query('reachSet')
        assert result is R
    
    def test_query_timePoint(self):
        """Test query method for timePoint"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2)), Zonotope([2, 3], np.eye(2))],
            'time': [0.0, 0.1]
        }
        R = ReachSet(timePoint)
        
        # Query without time value
        result = R.query('timePoint')
        assert result == timePoint
        
        # Query with specific time value
        result = R.query('timePoint', 0.05)
        assert isinstance(result, Zonotope)
    
    def test_query_timeInterval(self):
        """Test query method for timeInterval"""
        timeInterval = {
            'set': [Zonotope([1, 2], np.eye(2) * 0.1)],
            'time': [[0.0, 0.1]]
        }
        R = ReachSet({}, timeInterval)
        
        result = R.query('timeInterval')
        assert result == timeInterval
    
    def test_query_invalid_property(self):
        """Test query method with invalid property"""
        R = ReachSet()
        
        with pytest.raises(ValueError, match="Unknown property"):
            R.query('invalid_property')
    
    def test_project_timePoint(self):
        """Test project method for timePoint sets"""
        # Create 3D zonotope
        center = np.array([1, 2, 3])
        generators = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
        zono = Zonotope(center, generators)
        
        timePoint = {
            'set': [zono],
            'time': [0.0],
            'error': [0.01]
        }
        R = ReachSet(timePoint)
        
        # Project to first two dimensions
        R_proj = R.project([0, 1])
        
        assert len(R_proj.timePoint['set']) == 1
        assert R_proj.timePoint['time'] == [0.0]
        assert R_proj.timePoint['error'] == [0.01]
        
        # Check that the projected set has correct dimensions
        proj_set = R_proj.timePoint['set'][0]
        assert isinstance(proj_set, Zonotope)
        assert proj_set.c.shape[0] == 2  # 2D after projection
    
    def test_project_timeInterval(self):
        """Test project method for timeInterval sets"""
        # Create 3D zonotope
        center = np.array([1, 2, 3])
        generators = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
        zono = Zonotope(center, generators)
        
        timeInterval = {
            'set': [zono],
            'time': [[0.0, 0.1]]
        }
        R = ReachSet({}, timeInterval)
        
        # Project to last two dimensions
        R_proj = R.project([1, 2])
        
        assert len(R_proj.timeInterval['set']) == 1
        assert R_proj.timeInterval['time'] == [[0.0, 0.1]]
        
        # Check that the projected set has correct dimensions
        proj_set = R_proj.timeInterval['set'][0]
        assert isinstance(proj_set, Zonotope)
        assert proj_set.c.shape[0] == 2  # 2D after projection
    
    def test_project_numpy_arrays(self):
        """Test project method with numpy arrays"""
        timePoint = {
            'set': [np.array([1, 2, 3, 4])],
            'time': [0.0]
        }
        R = ReachSet(timePoint)
        
        # Project to first two dimensions
        R_proj = R.project([0, 1])
        
        projected_array = R_proj.timePoint['set'][0]
        expected = np.array([1, 2])
        assert np.allclose(projected_array, expected)
    
    def test_str_representation(self):
        """Test string representation"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2)), Zonotope([2, 3], np.eye(2))],
            'time': [0.0, 0.1]
        }
        timeInterval = {
            'set': [Zonotope([1, 2], np.eye(2) * 0.1)],
            'time': [[0.0, 0.1]]
        }
        R = ReachSet(timePoint, timeInterval)
        
        str_repr = str(R)
        assert "reachSet" in str_repr
        assert "time-point sets: 2" in str_repr
        assert "time-interval sets: 1" in str_repr
    
    def test_contains_not_implemented(self):
        """Test that contains method raises NotImplementedError"""
        R = ReachSet()
        
        with pytest.raises(NotImplementedError):
            R.contains(None)


if __name__ == "__main__":
    pytest.main([__file__]) 