import pytest
from unittest.mock import patch
import numpy as np
from cora_python.contSet import Interval

@patch('cora_python.contSet.interval.affine.Affine')
def test_affine_conversion(mock_affine):
    # Create an interval
    I = Interval(np.array([-1, 1]), np.array([1, 2]))

    # Call the affine conversion function
    I.affine('test_name', 'int')

    # Check if the Affine constructor was called once
    assert mock_affine.call_count == 1
    
    # Get the call arguments
    call_args = mock_affine.call_args[0]
    call_kwargs = mock_affine.call_args[1] if mock_affine.call_args[1] else {}
    
    # Check the arguments manually (handle numpy arrays properly)
    expected_inf = I.infimum()
    expected_sup = I.supremum()
    
    assert len(call_args) == 4
    assert np.allclose(call_args[0], expected_inf)
    assert np.allclose(call_args[1], expected_sup)
    assert call_args[2] == 'test_name'
    assert call_args[3] == 'int' 