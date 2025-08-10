import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid


def test_and_outer_ellipsoid_ellipsoid():
    Q1 = np.array([[3.0, -1.0], [-1.0, 1.5]])
    q1 = np.array([[0.5], [0.2]])
    E1 = Ellipsoid(Q1, q1)

    Q2 = np.array([[2.0, 0.3], [0.3, 1.2]])
    q2 = np.array([[0.2], [0.0]])
    E2 = Ellipsoid(Q2, q2)

    E_out = E1.and_(E2, 'outer')
    assert isinstance(E_out, Ellipsoid)
    assert E_out.Q.shape == (2, 2)


def test_and_inner_ellipsoid_ellipsoid():
    Q1 = np.array([[2.0, 0.1], [0.1, 1.5]])
    q1 = np.array([[0.1], [0.1]])
    E1 = Ellipsoid(Q1, q1)

    Q2 = np.array([[1.5, 0.0], [0.0, 1.0]])
    q2 = np.array([[0.0], [0.0]])
    E2 = Ellipsoid(Q2, q2)

    E_in = E1.and_(E2, 'inner')
    assert isinstance(E_in, Ellipsoid)
    # Inner approx should not be empty when outer says they intersect
    E_out = E1.and_(E2, 'outer')
    if not E_out.isemptyobject():
        assert not E_in.isemptyobject()

