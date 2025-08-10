import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_containsPoint import priv_containsPoint


def test_priv_containsPoint_basic():
    E = Ellipsoid(np.eye(2), np.zeros((2,1)))
    p_inside = np.array([[0.1],[0.2]])
    p_outside = np.array([[2.0],[0.0]])
    res_in, cert_in, _ = priv_containsPoint(E, p_inside, 1e-10)
    res_out, cert_out, _ = priv_containsPoint(E, p_outside, 1e-10)
    assert bool(res_in[0]) is True and bool(res_out[0]) is False


