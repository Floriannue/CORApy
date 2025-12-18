"""
test_spectraShadow - unit test function of spectraShadow (constructor)

Syntax:
    res = test_spectraShadow

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Adrian Kulmburg
Written:       ---
Last update:   ---
Last revision: ---
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestSpectraShadow:
    """Test class for spectraShadow constructor"""
    
    def test_constructor_syntaxes(self):
        """Test different constructor syntaxes"""
        # 2 dimensional box with radius 3 around point [-1;2]:
        A0 = np.eye(3)
        A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        
        # Only A-matrix
        S = SpectraShadow([A0, A1, A2])
        
        # Only A and c
        S = SpectraShadow([A0, A1, A2], np.array([[-1], [2]]))
        
        # A, c, and G
        S = SpectraShadow([A0, A1, A2], np.array([[-1], [2]]), 3 * np.eye(2))
        
        # Initialization through ESumRep (cell array with 2 elements in MATLAB)
        # In MATLAB, [A0,A1,A2] horizontally concatenates into a single matrix
        S = SpectraShadow([np.hstack([A0, A1, A2]), np.eye(3)])
    
    def test_wrong_initializations(self):
        """Test wrong initializations"""
        # Wrong initializations
        A = np.array([[1, 0, -1]])
        A_nonSymmetric = A.T
        ESumRep_ = [A]  # List (cell array in MATLAB) with 1 element - should fail
        c = np.array([[0], [1]])
        c_ = np.array([[3], [2], [3], [2]])
        G_ = np.eye(3)
        
        # Dimension mismatch
        with pytest.raises(CORAerror) as exc_info:
            SpectraShadow(A, c_)
        assert exc_info.value.identifier == 'CORA:wrongInputInConstructor'
        
        with pytest.raises(CORAerror) as exc_info:
            SpectraShadow(A, c, G_)
        assert exc_info.value.identifier == 'CORA:wrongInputInConstructor'
        
        # Incorrect ESumRep structure
        with pytest.raises(CORAerror) as exc_info:
            SpectraShadow(ESumRep_)
        assert exc_info.value.identifier == 'CORA:wrongInputInConstructor'
        
        # Empty argument
        with pytest.raises(CORAerror) as exc_info:
            SpectraShadow(np.array([]).reshape(0, 0), c)
        assert exc_info.value.identifier == 'CORA:wrongValue'
        
        # Too many arguments
        with pytest.raises(CORAerror) as exc_info:
            SpectraShadow(np.array([]).reshape(0, 0), A, c, G_, c, c)
        assert exc_info.value.identifier == 'CORA:numInputArgsConstructor' 