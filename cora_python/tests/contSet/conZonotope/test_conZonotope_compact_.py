import numpy as np

from cora_python.contSet.conZonotope import ConZonotope


def test_conZonotope_compact_zeros_generated():
    """Generated test: remove zero constraints and generators."""
    c = np.array([[0.0], [0.0]])
    G = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    A = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    b = np.array([[0.0], [1.0], [0.0]])
    cZ = ConZonotope(c, G, A, b)

    cZ_compact = cZ.compact_('zeros', 1e-12)

    assert np.allclose(cZ_compact.G, np.array([[1.0], [0.0]]))
    assert np.allclose(cZ_compact.A, np.array([[1.0]]))
    assert np.allclose(cZ_compact.b, np.array([[1.0]]))
