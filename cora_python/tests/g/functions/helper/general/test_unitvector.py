import numpy as np
import pytest
from cora_python.g.functions.helper.general.unitvector import unitvector
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class TestUnitvector:
    def test_basic_cases(self):
        # 1-based indexing test
        v1 = unitvector(1, 3)
        assert np.array_equal(v1, np.array([[1.], [0.], [0.]]))

        v2 = unitvector(2, 3)
        assert np.array_equal(v2, np.array([[0.], [1.], [0.]]))

        v3 = unitvector(3, 3)
        assert np.array_equal(v3, np.array([[0.], [0.], [1.]]))

    def test_single_dimension(self):
        v_single = unitvector(1, 1)
        assert np.array_equal(v_single, np.array([[1.]]))

    def test_invalid_index_high(self):
        with pytest.raises(CORAerror) as excinfo:
            unitvector(4, 3)
        assert 'Index must be a positive integer within the dimension.' in str(excinfo.value)

    def test_invalid_index_low(self):
        with pytest.raises(CORAerror) as excinfo:
            unitvector(0, 3)
        assert 'Index must be a positive integer within the dimension.' in str(excinfo.value)

    def test_invalid_dimension_low(self):
        with pytest.raises(CORAerror) as excinfo:
            unitvector(1, 0)
        assert 'Dimension must be a positive integer.' in str(excinfo.value)

    def test_invalid_dimension_negative(self):
        with pytest.raises(CORAerror) as excinfo:
            unitvector(1, -5)
        assert 'Dimension must be a positive integer.' in str(excinfo.value)

    def test_non_integer_inputs(self):
        with pytest.raises(CORAerror):
            unitvector(1.5, 3)
        with pytest.raises(CORAerror):
            unitvector(1, 3.0) 