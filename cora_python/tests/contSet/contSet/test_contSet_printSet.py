import pytest
import numpy as np

from cora_python.contSet.zonotope.zonotope import Zonotope

def test_contSet_printSet(capsys):
    # Create a simple zonotope for testing
    c = np.array([1, 2])
    G = np.array([[1, 0.5], [0.5, 1]])
    Z = Zonotope(c, G)

    # Test 1: Default call
    # The main goal is to ensure it runs without error
    Z.printSet()
    captured = capsys.readouterr()
    assert "Zonotope" in captured.out
    assert "np.array" in captured.out

    # Test 2: High accuracy
    Z.printSet('high')
    captured = capsys.readouterr()
    assert "Zonotope" in captured.out
    assert "np.array" in captured.out

    # Test 3: Compact format
    Z.printSet('%4.3f', True)
    captured = capsys.readouterr()
    assert "Zonotope" in captured.out
    
    # Test 4: Different accuracy
    Z.printSet('%2.1f')
    captured = capsys.readouterr()
    assert "Zonotope" in captured.out

    # Test 5: Call on a nested property (to exercise recursion)
    # Create a dummy object with a zonotope property
    class MockObject:
        def __init__(self, z):
            self.z = z
        
        def getPrintSetInfo(self):
            return "Mock", ['z']
    
    # Test that the function works without error on a mock object
    mock = MockObject(Z)
    from cora_python.contSet.contSet.printSet import printSet
    printSet(mock, 'high')
    captured = capsys.readouterr()
    assert "Mock" in captured.out 