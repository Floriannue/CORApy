# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.origin import origin
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class TestOrigin:

    def test_origin_2d(self):
        # Test 2D origin polytope
        P = origin(2)
        
        # Check that it's a 2D polytope
        assert P.dim() == 2
        
        # Check vertex representation - should contain only the origin
        # In d × n_vertices format: (2, 1) for 1 vertex in 2D space
        assert P._V.shape == (2, 1)
        assert np.array_equal(P._V, np.array([[0], [0]]))
        assert P._has_v_rep == True
        
        # Check halfspace representation
        expected_A = np.array([[1, 0], [0, 1], [-1, -1]])
        expected_b = np.array([[0], [0], [0]])
        assert np.array_equal(P.A, expected_A)
        assert np.array_equal(P.b, expected_b)

    def test_origin_3d(self):
        # Test 3D origin polytope
        P = origin(3)
        
        # Check that it's a 3D polytope
        assert P.dim() == 3
        
        # Check vertex representation - should contain only the origin
        # In d × n_vertices format: (3, 1) for 1 vertex in 3D space
        assert P._V.shape == (3, 1)
        assert np.array_equal(P._V, np.array([[0], [0], [0]]))
        assert P._has_v_rep == True
        
        # Check halfspace representation
        expected_A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]])
        expected_b = np.array([[0], [0], [0], [0]])
        assert np.array_equal(P.A, expected_A)
        assert np.array_equal(P.b, expected_b)

    def test_origin_invalid_input(self):
        # Test invalid input (non-positive dimension)
        with pytest.raises(CORAError) as excinfo:
            origin(0)
        assert "CORA:wrongInput" in str(excinfo.value)

        with pytest.raises(CORAError) as excinfo:
            origin(-1)
        assert "CORA:wrongInput" in str(excinfo.value)

        with pytest.raises(CORAError) as excinfo:
            origin(1.5)
        assert "CORA:wrongInput" in str(excinfo.value) 