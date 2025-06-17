"""
test_reachSet - comprehensive unit test function for reachSet class

Tests the reachSet class for storing reachable sets with all methods.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch
# Set matplotlib backend before importing anything that might use it
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from cora_python.g.classes.reachSet import ReachSet
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestReachSet:
    def test_reachSet_constructor_empty(self):
        """Test empty constructor"""
        R = ReachSet()
        assert len(R.timePoint.keys()) == 0
        assert len(R.timeInterval.keys()) == 0
        assert R.loc == 0
        assert R.parent == 0
    
    def test_reachSet_constructor_timePoint_only(self):
        """Test constructor with only timePoint"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2))],
            'time': [0.0]
        }
        R = ReachSet(timePoint)
        assert R.timePoint['set'] == timePoint['set']
        assert R.timePoint['time'] == timePoint['time']
        assert len(R.timeInterval.keys()) == 0
    
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
        assert R.timePoint['set'] == timePoint['set']
        assert R.timePoint['time'] == timePoint['time']
        assert R.timeInterval['set'] == timeInterval['set']
        assert R.timeInterval['time'] == timeInterval['time']
    
    def test_reachSet_constructor_with_location(self):
        """Test constructor with location"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2))],
            'time': [0.0]
        }
        R = ReachSet(timePoint, {}, 2, 5)  # parent=2, loc=5
        assert R.parent == 2
        assert R.loc == 5
    
    def test_initReachSet_static_method(self):
        """Test static initReachSet method"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2))],
            'time': [0.0]
        }
        R = ReachSet.initReachSet(timePoint)
        assert isinstance(R, ReachSet)
        assert R.timePoint['set'] == timePoint['set']
        assert R.timePoint['time'] == timePoint['time']
        assert len(R.timeInterval.keys()) == 0
    
    def test_query_reachSet(self):
        """Test query method for reachSet"""
        timeInterval = {
            'set': [Zonotope([1, 2], np.eye(2))],
            'time': [[0.0, 0.1]]
        }
        R = ReachSet({}, timeInterval)
        
        result = R.query('reachSet')
        assert result == timeInterval['set']
    
    def test_query_timePoint(self):
        """Test query method for timePoint"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2)), Zonotope([2, 3], np.eye(2))],
            'time': [0.0, 0.1]
        }
        R = ReachSet(timePoint)
        
        # Query reachSetTimePoint
        result = R.query('reachSetTimePoint')
        assert result == timePoint['set']
    
    def test_query_timeInterval(self):
        """Test query method for timeInterval"""
        timeInterval = {
            'set': [Zonotope([1, 2], np.eye(2) * 0.1)],
            'time': [[0.0, 0.1]]
        }
        R = ReachSet({}, timeInterval)
        
        result = R.query('reachSet')  # This queries time-interval sets
        assert result == timeInterval['set']
    
    def test_query_tFinal(self):
        """Test query method for tFinal"""
        timeInterval = {
            'set': [Zonotope([1, 2], np.eye(2) * 0.1)],
            'time': [[0.0, 0.5]]
        }
        R = ReachSet({}, timeInterval)
        
        result = R.query('tFinal')
        assert result == 0.5
    
    def test_query_invalid_property(self):
        """Test query method with invalid property"""
        R = ReachSet()
        
        with pytest.raises(ValueError, match="Property must be one of"):
            R.query('invalid_property')
    
    def test_find_method(self):
        """Test find method"""
        timePoint = {
            'set': [Zonotope([1, 2], np.eye(2))],
            'time': [0.0]
        }
        R = ReachSet(timePoint)
        
        # Test find with time
        result = R.find('time', 0.0)
        assert isinstance(result, ReachSet)
        # The result should be a new reachSet with the matching time point
        assert len(result.timePoint['set']) == 1
        assert result.timePoint['time'][0] == 0.0
    
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
    
    def test_add_method(self):
        """Test add method"""
        R1 = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        R2 = ReachSet({'set': [Zonotope([2, 3], np.eye(2))], 'time': [0.1]})
        
        R_sum = R1.add(R2)
        assert len(R_sum.timePoint['set']) == 2
        assert len(R_sum.timePoint['time']) == 2
    
    def test_append_method(self):
        """Test append method"""
        R1 = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        R2 = ReachSet({'set': [Zonotope([2, 3], np.eye(2))], 'time': [0.1]})
        
        R1.append(R2)
        assert len(R1.timePoint['set']) == 2
        assert len(R1.timePoint['time']) == 2
    
    def test_children_method(self):
        """Test children method"""
        R = [ReachSet(), ReachSet(), ReachSet()]
        R[1].parent = 0
        R[2].parent = 0
        
        children = R[0].children(R, 0)
        assert len(children) == 2
        assert 1 in children
        assert 2 in children
    
    def test_contains_method(self):
        """Test contains method"""
        R = ReachSet()
        
        # Should raise NotImplementedError for now
        with pytest.raises(NotImplementedError):
            R.contains(np.array([1, 2]))
    
    def test_isequal_method(self):
        """Test isequal method"""
        timePoint = {'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]}
        R1 = ReachSet(timePoint)
        R2 = ReachSet(timePoint)
        R3 = ReachSet({'set': [Zonotope([2, 3], np.eye(2))], 'time': [0.1]})
        
        assert R1.isequal(R2)
        assert not R1.isequal(R3)
    
    def test_isemptyobject_method(self):
        """Test isemptyobject method"""
        R_empty = ReachSet()
        R_nonempty = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        
        assert R_empty.isemptyobject()
        assert not R_nonempty.isemptyobject()
    
    def test_order_method(self):
        """Test order method"""
        R = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        
        # Should return the order of the first set
        order_result = R.order()
        assert isinstance(order_result, (int, float))
    
    def test_arithmetic_operations(self):
        """Test arithmetic operations"""
        R1 = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        R2 = ReachSet({'set': [Zonotope([2, 3], np.eye(2))], 'time': [0.0]})
        
        # Test plus
        R_plus = R1.plus(R2)
        assert isinstance(R_plus, ReachSet)
        
        # Test minus
        R_minus = R1.minus(R2)
        assert isinstance(R_minus, ReachSet)
        
        # Test times
        R_times = R1.times(2)
        assert isinstance(R_times, ReachSet)
        
        # Test mtimes
        R_mtimes = R1.mtimes(np.eye(2))
        assert isinstance(R_mtimes, ReachSet)
    
    def test_unary_operations(self):
        """Test unary operations"""
        R = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        
        # Test uminus
        R_uminus = R.uminus()
        assert isinstance(R_uminus, ReachSet)
        
        # Test uplus
        R_uplus = R.uplus()
        assert isinstance(R_uplus, ReachSet)
    
    def test_comparison_operations(self):
        """Test comparison operations"""
        R1 = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        R2 = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        R3 = ReachSet({'set': [Zonotope([2, 3], np.eye(2))], 'time': [0.0]})
        
        # Test eq
        assert R1.eq(R2)
        assert not R1.eq(R3)
        
        # Test ne
        assert not R1.ne(R2)
        assert R1.ne(R3)
    
    def test_shiftTime_method(self):
        """Test shiftTime method"""
        timePoint = {'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]}
        R = ReachSet(timePoint)
        
        R_shifted = R.shiftTime(1.0)
        assert R_shifted.timePoint['time'][0] == 1.0
    
    def test_plot_method(self):
        """Test plot method"""
        R = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        
        # Should not raise an error
        with patch('matplotlib.pyplot.show'):
            R.plot()
    
    def test_plotOverTime_method(self):
        """Test plotOverTime method"""
        R = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        
        # Should not raise an error
        with patch('matplotlib.pyplot.show'):
            R.plotOverTime()
    
    def test_plotTimeStep_method(self):
        """Test plotTimeStep method"""
        timeInterval = {
            'set': [Zonotope([1, 2], np.eye(2))],
            'time': [[0.0, 0.1]]
        }
        R = ReachSet({}, timeInterval)
        
        # Should not raise an error
        with patch('matplotlib.pyplot.show'):
            R.plotTimeStep()
    
    def test_plotAsGraph_method(self):
        """Test plotAsGraph method"""
        R = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        
        # Should not raise an error
        with patch('matplotlib.pyplot.show'):
            R.plotAsGraph()
    
    def test_modelChecking_method(self):
        """Test modelChecking method"""
        R = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        
        # Create a mock STL formula with required attributes
        mock_formula = Mock()
        mock_formula.time_bound = 1.0
        mock_formula.variables = ['x1', 'x2']
        mock_formula.time = 1.0
        mock_formula.disjunction = Mock()
        mock_formula.disjunction.time = 1.0
        
        # Test different algorithms
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result1 = R.modelChecking(mock_formula, 'sampledTime')
            assert isinstance(result1, bool)
            
            result2 = R.modelChecking(mock_formula, 'rtl')
            assert isinstance(result2, bool)
            
            result3 = R.modelChecking(mock_formula, 'signals')
            assert isinstance(result3, bool)
            
            result4 = R.modelChecking(mock_formula, 'incremental')
            assert isinstance(result4, bool)
    
    def test_modelChecking_invalid_algorithm(self):
        """Test modelChecking with invalid algorithm"""
        R = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]})
        mock_formula = Mock()
        
        with pytest.raises(ValueError, match="Algorithm must be one of"):
            R.modelChecking(mock_formula, 'invalid')
    
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
    
    def test_empty_reachSet_operations(self):
        """Test operations on empty reachSet"""
        R = ReachSet()
        
        # Test various operations on empty reachSet
        assert R.isemptyobject()
        
        # Query operations should handle empty case
        result = R.query('reachSetTimePoint')
        assert result == []
        
        # Projection should return empty reachSet
        R_proj = R.project([0, 1])
        assert R_proj.isemptyobject()
    
    def test_multiple_locations(self):
        """Test reachSet with multiple locations"""
        R1 = ReachSet({'set': [Zonotope([1, 2], np.eye(2))], 'time': [0.0]}, {}, 0, 1)  # parent=0, loc=1
        R2 = ReachSet({'set': [Zonotope([2, 3], np.eye(2))], 'time': [0.1]}, {}, 0, 2)  # parent=0, loc=2
        
        assert R1.loc == 1
        assert R2.loc == 2
        
        # Test addition preserves locations
        R_sum = R1.add(R2)
        assert hasattr(R_sum, 'loc')
    
    def test_error_handling(self):
        """Test error handling in various methods"""
        R = ReachSet()
        
        # Test invalid inputs
        with pytest.raises((ValueError, TypeError)):
            R.project("invalid")
        
        with pytest.raises((ValueError, TypeError)):
            R.add("invalid")


if __name__ == "__main__":
    pytest.main([__file__]) 