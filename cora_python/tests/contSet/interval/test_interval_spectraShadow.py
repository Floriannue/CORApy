import pytest
import numpy as np
from scipy.sparse import csc_matrix
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow

def test_interval_spectraShadow():
    # Example from MATLAB documentation
    I = Interval(np.array([[-3.14], [-4.20]]), np.array([[2.718], [0.69]]))
    SpS = I.spectraShadow()

    assert isinstance(SpS, SpectraShadow)
    
    # Manually construct expected A matrix
    # A = [A0, A1, A2]
    A0_diag1 = np.array([[2.718, 0], [0, 3.14]])
    A0_diag2 = np.array([[0.69, 0], [0, 4.20]])
    A0 = np.block([[A0_diag1, np.zeros((2,2))], [np.zeros((2,2)), A0_diag2]])

    A1 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    A2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    expected_A = np.hstack((A0, A1, A2))
    
    # Compare the generated matrix
    assert np.allclose(SpS.A.toarray(), expected_A)
    
    # Compare the copied properties
    assert SpS.bounded == I.is_bounded()
    assert SpS.emptySet == I.is_empty()
    assert SpS.fullDim == I.isFullDim()
    assert np.allclose(SpS.center, I.center()) 