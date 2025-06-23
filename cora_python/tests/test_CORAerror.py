"""
Test suite for CORAerror class

This module tests all functionality of the CORAerror exception class,
ensuring it properly handles all error types defined in the MATLAB version.

Authors: Python translation test suite
Written: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestCORAerror:
    """Test class for CORAerror functionality"""
    
    def test_basic_error_creation(self):
        """Test basic error creation and message generation"""
        error = CORAerror('CORA:specialError', 'Test message')
        assert error.identifier == 'CORA:specialError'
        assert error.message == 'Test message'
        assert 'Test message' in str(error)
    
    def test_wrong_input_in_constructor(self):
        """Test CORA:wrongInputInConstructor error type"""
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:wrongInputInConstructor', 'Invalid arguments provided')
        
        error = exc_info.value
        assert 'Wrong input arguments for constructor' in str(error)
        assert 'Invalid arguments provided' in str(error)
        assert 'Type \'help' in str(error)
    
    def test_no_input_in_set_constructor(self):
        """Test CORA:noInputInSetConstructor error type"""
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:noInputInSetConstructor')
        
        error = exc_info.value
        assert 'No input arguments for constructor' in str(error)
        assert 'Please consider calling' in str(error)
        assert '.empty or' in str(error)
        assert '.Inf instead' in str(error)
    
    def test_dimension_mismatch_with_objects(self):
        """Test CORA:dimensionMismatch error type with mock objects"""
        class MockObject1:
            def dim(self):
                return 3
        
        class MockObject2:
            def dim(self):
                return 5
        
        obj1 = MockObject1()
        obj2 = MockObject2()
        
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:dimensionMismatch', '', obj1, obj2)
        
        error = exc_info.value
        assert 'MockObject1' in str(error)
        assert 'MockObject2' in str(error)
        assert 'dimension/size 3' in str(error)
        assert 'dimension/size 5' in str(error)
    
    def test_dimension_mismatch_with_arrays(self):
        """Test CORA:dimensionMismatch error type with numpy arrays"""
        arr1 = np.array([[1, 2], [3, 4]])  # 2x2
        arr2 = np.array([1, 2, 3])  # (3,)
        
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:dimensionMismatch', '', arr1, arr2)
        
        error = exc_info.value
        assert 'ndarray' in str(error)
        assert '(2, 2)' in str(error)
        assert '(3,)' in str(error)
    
    def test_empty_set(self):
        """Test CORA:emptySet error type"""
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:emptySet')
        
        error = exc_info.value
        assert str(error) == 'Set is empty!'
    
    def test_file_not_found(self):
        """Test CORA:fileNotFound error type"""
        filename = 'nonexistent_file.txt'
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:fileNotFound', filename)
        
        error = exc_info.value
        assert f'File with name {filename} could not be found.' in str(error)
    
    def test_wrong_value_regular_argument(self):
        """Test CORA:wrongValue error type for regular arguments"""
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:wrongValue', 'first input argument', 'positive number')
        
        error = exc_info.value
        assert 'Wrong value for the first input argument input argument' in str(error)
        assert 'The right value: positive number' in str(error)
        assert 'Type \'help' in str(error)
    
    def test_wrong_value_name_value_pair(self):
        """Test CORA:wrongValue error type for name-value pairs"""
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:wrongValue', 'name-value pair \'method\'', 'exact')
        
        error = exc_info.value
        assert 'Wrong value for name-value pair \'method\'' in str(error)
        assert 'The right value: exact' in str(error)
    
    def test_plot_properties(self):
        """Test CORA:plotProperties error type"""
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:plotProperties', 'Custom plot error message')
        
        error = exc_info.value
        assert str(error) == 'Custom plot error message'
    
    def test_plot_properties_empty(self):
        """Test CORA:plotProperties error type with empty message"""
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:plotProperties')
        
        error = exc_info.value
        assert 'Incorrect plotting properties specified.' in str(error)
    
    def test_not_supported(self):
        """Test CORA:notSupported error type"""
        message = 'This operation is not supported for the given input types'
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:notSupported', message)
        
        error = exc_info.value
        assert str(error) == message
    
    def test_not_defined(self):
        """Test CORA:notDefined error type"""
        message = 'Matrix multiplication for these set types'
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:notDefined', message)
        
        error = exc_info.value
        assert f'Undefined functionality: {message}' in str(error)
    
    def test_special_error(self):
        """Test CORA:specialError error type"""
        message = 'A very specific error occurred'
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:specialError', message)
        
        error = exc_info.value
        assert str(error) == message
    
    def test_noops_with_objects(self):
        """Test CORA:noops error type with multiple objects"""
        class MockClass1:
            pass
        
        class MockClass2:
            pass
        
        obj1 = MockClass1()
        obj2 = MockClass2()
        
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:noops', '', obj1, obj2)
        
        error = exc_info.value
        assert 'The function' in str(error)
        assert 'is not implemented for the following arguments' in str(error)
        assert 'MockClass1, MockClass2' in str(error)
        assert 'Type \'help' in str(error)
    
    def test_noops_no_objects(self):
        """Test CORA:noops error type without objects"""
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:noops')
        
        error = exc_info.value
        assert 'The function' in str(error)
        assert 'is not implemented for the given arguments' in str(error)
    
    def test_no_exact_alg_with_objects(self):
        """Test CORA:noExactAlg error type with objects"""
        class MockClass1:
            pass
        
        obj1 = MockClass1()
        
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:noExactAlg', '', obj1)
        
        error = exc_info.value
        assert 'There is no exact algorithm for function' in str(error)
        assert 'MockClass1' in str(error)
    
    def test_no_exact_alg_no_objects(self):
        """Test CORA:noExactAlg error type without objects"""
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:noExactAlg')
        
        error = exc_info.value
        assert 'There is no exact algorithm for function' in str(error)
    
    def test_solver_issue_with_solver_name(self):
        """Test CORA:solverIssue error type with solver name"""
        solver_name = 'MOSEK'
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:solverIssue', solver_name)
        
        error = exc_info.value
        assert f'Solver ({solver_name})' in str(error)
        assert 'failed due to numerical/other issues' in str(error)
        assert 'Type \'help' in str(error)
    
    def test_solver_issue_no_solver_name(self):
        """Test CORA:solverIssue error type without solver name"""
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:solverIssue')
        
        error = exc_info.value
        assert 'Solver in' in str(error)
        assert 'failed due to numerical/other issues' in str(error)
    
    def test_out_of_domain(self):
        """Test CORA:outOfDomain error type"""
        message = 'Input must be positive'
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:outOfDomain', message)
        
        error = exc_info.value
        assert 'Input is not inside the valid domain' in str(error)
        assert message in str(error)
        assert 'Type \'help' in str(error)
    
    def test_unknown_identifier(self):
        """Test unknown error identifier"""
        unknown_id = 'CORA:unknownError'
        message = 'This is an unknown error'
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror(unknown_id, message)
        
        error = exc_info.value
        assert f'{unknown_id}: {message}' in str(error)
    
    def test_caller_info_extraction(self):
        """Test that caller information is properly extracted"""
        def dummy_function():
            raise CORAerror('CORA:specialError', 'Test from function')
        
        with pytest.raises(CORAerror) as exc_info:
            dummy_function()
        
        error = exc_info.value
        # Note: pytest wraps function calls, so we check that some function name is extracted
        assert error.functionname is not None
        assert len(error.functionname) > 0
        assert error.filename is not None
        assert error.classname is not None
    
    def test_convenience_function(self):
        """Test the CORAerror convenience function"""
        with pytest.raises(CORAerror) as exc_info:
            CORAerror('CORA:specialError', 'Test convenience function')
        
        error = exc_info.value
        assert error.identifier == 'CORA:specialError'
        assert 'Test convenience function' in str(error)
    
    def test_convenience_function_no_message(self):
        """Test the CORAerror convenience function without message"""
        with pytest.raises(CORAerror) as exc_info:
            CORAerror('CORA:emptySet')
        
        error = exc_info.value
        assert error.identifier == 'CORA:emptySet'
        assert str(error) == 'Set is empty!'
    
    def test_error_attributes(self):
        """Test that error attributes are properly set"""
        error = CORAerror('CORA:specialError', 'Test message')
        
        assert hasattr(error, 'identifier')
        assert hasattr(error, 'message')
        assert hasattr(error, 'filename')
        assert hasattr(error, 'classname')
        assert hasattr(error, 'functionname')
        
        assert error.identifier == 'CORA:specialError'
        assert error.message == 'Test message'
    
    def test_error_inheritance(self):
        """Test that CORAerror properly inherits from Exception"""
        error = CORAerror('CORA:specialError', 'Test message')
        assert isinstance(error, Exception)
        assert isinstance(error, CORAerror)
    
    def test_multiple_args_handling(self):
        """Test handling of multiple arguments"""
        class MockObj1:
            pass
        
        class MockObj2:
            pass
        
        obj1, obj2 = MockObj1(), MockObj2()
        
        error = CORAerror('CORA:dimensionMismatch', 'test', obj1, obj2)
        assert len(error.args_list) == 2
        assert error.args_list[0] is obj1
        assert error.args_list[1] is obj2


class TestCORAerrorIntegration:
    """Integration tests for CORAerror with other components"""
    
    def test_error_in_nested_function_calls(self):
        """Test error handling in nested function calls"""
        def level1():
            return level2()
        
        def level2():
            return level3()
        
        def level3():
            raise CORAerror('CORA:specialError', 'Deep nested error')
        
        with pytest.raises(CORAerror) as exc_info:
            level1()
        
        error = exc_info.value
        assert 'Deep nested error' in str(error)
        # Note: pytest wraps function calls, so we check that some function name is extracted
        assert error.functionname is not None
        assert len(error.functionname) > 0
    
    def test_error_with_complex_objects(self):
        """Test error handling with complex objects"""
        class ComplexMockObject:
            def __init__(self, name, dimension):
                self.name = name
                self._dim = dimension
            
            def dim(self):
                return self._dim
            
            def __str__(self):
                return f"{self.name}({self._dim}D)"
        
        obj1 = ComplexMockObject("Set1", 3)
        obj2 = ComplexMockObject("Set2", 5)
        
        with pytest.raises(CORAerror) as exc_info:
            raise CORAerror('CORA:dimensionMismatch', '', obj1, obj2)
        
        error = exc_info.value
        assert 'ComplexMockObject' in str(error)
        assert 'dimension/size 3' in str(error)
        assert 'dimension/size 5' in str(error)


if __name__ == '__main__':
    pytest.main([__file__])