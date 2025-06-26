"""
test_contDynamics - unit tests for ContDynamics base class

This module contains comprehensive unit tests for the ContDynamics base class
used as the foundation for all continuous dynamical systems.

Authors: Python translation by AI Assistant
Written: 2025
"""

import pytest
import warnings
import numpy as np
from cora_python.contDynamics.contDynamics.contDynamics import ContDynamics

# The 'ConcreteContDynamics' class is now available from conftest.py

class TestContDynamics:
    """Test class for ContDynamics"""
    
    def test_init_basic(self, ConcreteContDynamics):
        """Test basic initialization"""
        sys = ConcreteContDynamics(
            name="test_system",
            states=2,
            inputs=1,
            outputs=2,
            dists=1,
            noises=0
        )
        
        assert sys.name == "test_system"
        assert sys.nr_of_dims == 2
        assert sys.nr_of_inputs == 1
        assert sys.nr_of_outputs == 2
        assert sys.nr_of_disturbances == 1
        assert sys.nr_of_noises == 0
    
    def test_init_default_values(self, ConcreteContDynamics):
        """Test initialization with default values"""
        sys = ConcreteContDynamics()
        
        assert sys.name == ""
        assert sys.nr_of_dims == 0
        assert sys.nr_of_inputs == 0
        assert sys.nr_of_outputs == 0
        assert sys.nr_of_disturbances == 0
        assert sys.nr_of_noises == 0
    
    def test_init_partial_values(self, ConcreteContDynamics):
        """Test initialization with partial values"""
        sys = ConcreteContDynamics(name="partial", states=3, inputs=2)
        
        assert sys.name == "partial"
        assert sys.nr_of_dims == 3
        assert sys.nr_of_inputs == 2
        assert sys.nr_of_outputs == 0
        assert sys.nr_of_disturbances == 0
        assert sys.nr_of_noises == 0
    
    def test_init_invalid_name(self, ConcreteContDynamics):
        """Test initialization with invalid name"""
        with pytest.raises(TypeError):
            ConcreteContDynamics(name=123)
        
        with pytest.raises(TypeError):
            ConcreteContDynamics(name=None)
        
        with pytest.raises(TypeError):
            ConcreteContDynamics(name=[])
    
    def test_init_invalid_dimensions(self, ConcreteContDynamics):
        """Test initialization with invalid dimensions"""
        # Test negative values
        with pytest.raises(ValueError):
            ConcreteContDynamics(states=-1)
        
        with pytest.raises(ValueError):
            ConcreteContDynamics(inputs=-1)
        
        with pytest.raises(ValueError):
            ConcreteContDynamics(outputs=-1)
        
        with pytest.raises(ValueError):
            ConcreteContDynamics(dists=-1)
        
        with pytest.raises(ValueError):
            ConcreteContDynamics(noises=-1)
        
        # Test non-integer values
        with pytest.raises(ValueError):
            ConcreteContDynamics(states=1.5)
        
        with pytest.raises(ValueError):
            ConcreteContDynamics(inputs="invalid")
        
        with pytest.raises(ValueError):
            ConcreteContDynamics(outputs=None)
    
    def test_str_representation(self, ConcreteContDynamics):
        """Test string representation"""
        sys = ConcreteContDynamics(
            name="test_sys",
            states=3,
            inputs=2,
            outputs=1
        )
        
        str_repr = str(sys)
        assert isinstance(str_repr, str)
        assert "Continuous dynamics: 'test_sys'" in str_repr

        repr_repr = repr(sys)
        assert isinstance(repr_repr, str)
        assert "ConcreteContDynamics" in repr_repr
    
    def test_repr_representation(self, ConcreteContDynamics):
        """Test detailed string representation"""
        sys = ConcreteContDynamics(
            name="test_sys",
            states=3,
            inputs=2,
            outputs=1,
            dists=1,
            noises=0
        )
        
        repr_str = repr(sys)
        assert isinstance(repr_str, str)
        assert "ConcreteContDynamics" in repr_str
        assert "name='test_sys'" in repr_str
        assert "states=3" in repr_str
        assert "inputs=2" in repr_str
        assert "outputs=1" in repr_str
        assert "dists=1" in repr_str
        assert "noises=0" in repr_str
    
    def test_legacy_dim_property(self, ConcreteContDynamics):
        """Test legacy dim property with deprecation warning"""
        sys = ConcreteContDynamics(states=5)
        
        # Test getter with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dim_value = sys.dim
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)
            assert dim_value == 5
        
        # Test setter with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sys.dim = 7
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)
            assert sys.nr_of_dims == 7
    
    def test_legacy_nr_of_states_property(self, ConcreteContDynamics):
        """Test legacy nr_of_states property with deprecation warning"""
        sys = ConcreteContDynamics(states=4)
        
        # Test getter with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            states_value = sys.nr_of_states
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)
            assert states_value == 4
        
        # Test setter with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sys.nr_of_states = 6
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)
            assert sys.nr_of_dims == 6
    
    def test_edge_cases(self, ConcreteContDynamics):
        """Test edge cases"""
        # Test with zero dimensions
        sys = ConcreteContDynamics(
            name="zero_system",
            states=0,
            inputs=0,
            outputs=0,
            dists=0,
            noises=0
        )
        
        assert sys.nr_of_dims == 0
        assert sys.nr_of_inputs == 0
        assert sys.nr_of_outputs == 0
        assert sys.nr_of_disturbances == 0
        assert sys.nr_of_noises == 0
        
        # Test with large dimensions
        sys = ConcreteContDynamics(
            name="large_system",
            states=1000,
            inputs=500,
            outputs=200
        )
        
        assert sys.nr_of_dims == 1000
        assert sys.nr_of_inputs == 500
        assert sys.nr_of_outputs == 200
        
        # Test with empty name
        sys = ConcreteContDynamics(name="")
        assert sys.name == ""
        
        # Test with special characters in name
        sys = ConcreteContDynamics(name="system_123!@#")
        assert sys.name == "system_123!@#"
    
    def test_property_access(self, ConcreteContDynamics):
        """Test direct property access and modification"""
        sys = ConcreteContDynamics(
            name="test",
            states=2,
            inputs=1,
            outputs=2
        )
        
        # Test direct property access
        assert sys.nr_of_dims == 2
        assert sys.nr_of_inputs == 1
        assert sys.nr_of_outputs == 2
        
        # Test direct property modification
        sys.nr_of_dims = 5
        sys.nr_of_inputs = 3
        sys.nr_of_outputs = 4
        sys.name = "modified"
        
        assert sys.nr_of_dims == 5
        assert sys.nr_of_inputs == 3
        assert sys.nr_of_outputs == 4
        assert sys.name == "modified"
    
    def test_abstract_class_instantiation(self, ConcreteContDynamics):
        """Test that ContDynamics cannot be instantiated directly"""
        with pytest.raises(TypeError):
            ContDynamics()
    
    def test_inheritance(self, ConcreteContDynamics):
        """Test inheritance properties"""
        sys = ConcreteContDynamics(name="child", states=3)
        
        # Test isinstance relationships
        assert isinstance(sys, ContDynamics)
        assert isinstance(sys, ConcreteContDynamics)
        
        # Test class hierarchy
        assert issubclass(ConcreteContDynamics, ContDynamics)
    
    def test_equality_and_comparison(self, ConcreteContDynamics):
        """Test equality and comparison if implemented"""
        sys1 = ConcreteContDynamics(name="sys1", states=2, inputs=1)
        sys2 = ConcreteContDynamics(name="sys1", states=2, inputs=1)
        sys3 = ConcreteContDynamics(name="sys2", states=2, inputs=1)
        
        # Basic identity tests
        assert sys1 is sys1
        assert sys1 is not sys2
        
        # If equality is not implemented, they should not be equal
        # (unless __eq__ is specifically implemented)
        # This is just testing the default behavior
        assert sys1 != sys2  # Different objects
        assert sys1 != sys3  # Different names


def test_contDynamics_integration(ConcreteContDynamics):
    """Integration test for ContDynamics with realistic usage"""
    # Create a realistic system
    sys = ConcreteContDynamics(
        name="LinearSystem_2D",
        states=2,
        inputs=1,
        outputs=2,
        dists=0,
        noises=0
    )
    
    # Test all properties
    assert sys.name == "LinearSystem_2D"
    assert sys.nr_of_dims == 2
    assert sys.nr_of_inputs == 1
    assert sys.nr_of_outputs == 2
    assert sys.nr_of_disturbances == 0
    assert sys.nr_of_noises == 0
    
    # Test string representations
    str_repr = str(sys)
    repr_str = repr(sys)
    
    # Test string representation (display)
    assert "LinearSystem_2D" in str_repr
    assert "number of dimensions: 2" in str_repr
    assert "number of inputs: 1" in str_repr
    assert "number of outputs: 2" in str_repr
    
    # Test repr representation
    assert "LinearSystem_2D" in repr_str
    assert "states=2" in repr_str
    assert "inputs=1" in repr_str
    assert "outputs=2" in repr_str
    
    # Test legacy properties with warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Access legacy properties
        old_dim = sys.dim
        old_states = sys.nr_of_states
        
        # Should have generated warnings
        assert len(w) == 2
        assert all(issubclass(warning.category, DeprecationWarning) for warning in w)
        
        # Values should be correct
        assert old_dim == 2
        assert old_states == 2
    
    # Test modification
    sys.name = "ModifiedSystem"
    sys.nr_of_dims = 3
    sys.nr_of_inputs = 2
    
    assert sys.name == "ModifiedSystem"
    assert sys.nr_of_dims == 3
    assert sys.nr_of_inputs == 2
    
    print("ContDynamics integration test completed successfully")


if __name__ == '__main__':
    test = TestContDynamics()
    test.test_init_basic()
    test.test_init_default_values()
    test.test_init_partial_values()
    test.test_init_invalid_name()
    test.test_init_invalid_dimensions()
    test.test_str_representation()
    test.test_repr_representation()
    test.test_legacy_dim_property()
    test.test_legacy_nr_of_states_property()
    test.test_edge_cases()
    test.test_property_access()
    test.test_abstract_class_instantiation()
    test.test_inheritance()
    test.test_equality_and_comparison()
    
    # Run integration test
    test_contDynamics_integration()
    
    print("All ContDynamics tests passed!") 