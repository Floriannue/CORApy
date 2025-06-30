import pytest
import numpy as np

from cora_python.contSet.zonotope.zonotope import Zonotope  
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check import compareMatrices

def test_contSet_projectOnHyperplane():
    # Test 1: Based on the example in projectOnHyperplane.m
    c = np.array([2, 2])
    G = np.array([[1, -1], [0, 1]])
    Z = Zonotope(c, G)
    
    # Hyperplane: 1*x + 1*y = 1 -> x + y - 1 = 0
    Ae = np.array([[1, 1]])
    be = np.array([1])
    hyp = Polytope(A_eq=Ae, b_eq=be)
    
    Z_proj = Z.projectOnHyperplane(hyp)
    
    # Expected result calculation
    # c_norm = [1/sqrt(2); 1/sqrt(2)], d_norm = 1/sqrt(2)
    # A = I - c*c'
    # b = d*c
    c_h = np.array([[1], [1]])
    norm_c = np.linalg.norm(c_h)
    c_norm = c_h / norm_c
    d_norm = be[0] / norm_c
    
    A_proj = np.eye(2) - c_norm @ c_norm.T
    b_proj = (d_norm * c_norm).flatten()
    
    c_exp = A_proj @ c + b_proj
    G_exp = A_proj @ G
    
    assert isinstance(Z_proj, Zonotope)
    assert compareMatrices(Z_proj.c, c_exp, 1e-12)
    assert compareMatrices(Z_proj.G, G_exp, 1e-12)

    # Test 2: Error when input is not a hyperplane
    # A polytope that is not a hyperplane (e.g., a box)
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([1, 1, 1, 1])
    not_a_hyperplane = Polytope(A=A, b=b)
    
    with pytest.raises(CORAerror) as e:
        Z.projectOnHyperplane(not_a_hyperplane)
    assert "must represent a hyperplane" in str(e.value) 