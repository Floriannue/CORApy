import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid
from scipy.special import gamma

def test_volume_():
    # Test 1: Empty Ellipsoid
    E_empty = Ellipsoid.empty(2) # From MATLAB test
    assert E_empty.volume_() == 0.0

    # Test 2: MATLAB case E0 (degenerate ellipsoid)
    # E0 = ellipsoid([ 0.0000000000000000 0.0000000000000000 ; 0.0000000000000000 0.0000000000000000 ], [ 1.0986933635979599 ; -1.9884387759871638 ], 0.000001);
    # In Python, a zero Q matrix means volume should be 0.
    E0_Q = np.array([[0.0, 0.0], [0.0, 0.0]])
    E0_center = np.array([[1.0986933635979599], [-1.9884387759871638]])
    E0 = Ellipsoid(E0_Q, E0_center)
    assert np.isclose(E0.volume_(), 0.0, atol=1e-9)

    # Test 3: MATLAB case Ed1 (nearly degenerate ellipsoid)
    # Ed1 = ellipsoid([ 4.2533342807136076 0.6346400221575308 ; 0.6346400221575309 0.0946946398147988 ], [ -2.4653656883489115 ; 0.2717868749873985 ], 0.000001);
    Ed1_Q = np.array([[4.2533342807136076, 0.6346400221575308], [0.6346400221575309, 0.0946946398147988]])
    Ed1_center = np.array([[-2.4653656883489115], [0.2717868749873985]])
    Ed1 = Ellipsoid(Ed1_Q, Ed1_center)
    assert np.isclose(Ed1.volume_(), 0.0, atol=1e-6) # determinant is very close to zero

    # Test 4: MATLAB case E1
    # E1 = ellipsoid([ 5.4387811500952807 12.4977183618314545 ; 12.4977183618314545 29.6662117284481646 ], [ -0.7445068341257537 ; 3.5800647524843665 ], 0.000001);
    E1_Q = np.array([[5.4387811500952807, 12.4977183618314545], [12.4977183618314545, 29.6662117284481646]])
    E1_center = np.array([[-0.7445068341257537], [3.5800647524843665]])
    E1 = Ellipsoid(E1_Q, E1_center)
    n1 = E1.dim()
    E1_vol_expected = np.pi**(n1/2) / gamma(n1/2 + 1) * np.sqrt(np.linalg.det(E1_Q))
    assert np.isclose(E1.volume_(), E1_vol_expected)

    # Test 5: Unit ball (2D)
    E_unit_ball_2D = Ellipsoid(np.eye(2))
    n_unit_ball_2D = E_unit_ball_2D.dim()
    expected_vol_unit_ball_2D = np.pi**(n_unit_ball_2D/2) / gamma(n_unit_ball_2D/2 + 1) * np.sqrt(np.linalg.det(np.eye(2)))
    assert np.isclose(E_unit_ball_2D.volume_(), expected_vol_unit_ball_2D)
    
    # Test 6: Unit ball (3D)
    E_unit_ball_3D = Ellipsoid(np.eye(3))
    n_unit_ball_3D = E_unit_ball_3D.dim()
    expected_vol_unit_ball_3D = np.pi**(n_unit_ball_3D/2) / gamma(n_unit_ball_3D/2 + 1) * np.sqrt(np.linalg.det(np.eye(3)))
    assert np.isclose(E_unit_ball_3D.volume_(), expected_vol_unit_ball_3D) 