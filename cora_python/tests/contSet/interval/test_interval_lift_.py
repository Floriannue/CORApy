import numpy as np
import pytest
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class TestIntervalLift:
    def test_lift(self):
        I = Interval([1, 2], [3, 4])
        N = 4
        proj = np.array([1, 3])
        res = I.lift_(N, proj)

        expected_lb = -np.inf * np.ones((N, 1))
        expected_ub = np.inf * np.ones((N, 1))
        expected_lb[proj] = I.inf.reshape(-1, 1)
        expected_ub[proj] = I.sup.reshape(-1, 1)
        
        expected = Interval(expected_lb, expected_ub)

        assert res.isequal(expected)

    def test_lift_empty_error(self):
        I = Interval.empty()
        with pytest.raises(CORAerror) as e:
            I.lift_(4, np.array([1, 3]))
        assert "not supported for empty intervals" in str(e.value) 