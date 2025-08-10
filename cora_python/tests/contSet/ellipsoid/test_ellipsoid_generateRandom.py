from cora_python.contSet.ellipsoid import Ellipsoid


def test_generateRandom_basic():
    E = Ellipsoid.generateRandom('Dimension', 2)
    assert isinstance(E, Ellipsoid)
    assert E.Q.shape == (2, 2)
    assert E.q.shape == (2, 1)

