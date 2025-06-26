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
    assert "zonotope" in captured.out
    assert "c: [1 2]" in captured.out

    # Test 2: High accuracy
    Z.printSet('high')
    captured = capsys.readouterr()
    assert "zonotope" in captured.out
    assert "G: [[1.  0.5]" in captured.out

    # Test 3: With braces
    Z.printSet('high', use_braces=True)
    captured = capsys.readouterr()
    assert "{" in captured.out
    assert "}" in captured.out
    
    # Test 4: Without braces
    Z.printSet('low', use_braces=False)
    captured = capsys.readouterr()
    assert "{" not in captured.out
    assert "}" not in captured.out

    # Test 5: Call on a nested property (to exercise recursion)
    # Create a dummy object with a zonotope property
    class MockObject:
        def __init__(self, z):
            self.z = z
        
        def getPrintSetInfo(self):
            return "Mock", [('z', self.z)]
        
        def printSet(self, *args, **kwargs):
            # Temporarily attach the real printSet for the test
            from cora_python.contSet.contSet.printSet import printSet as ps
            return ps(self, *args, **kwargs)

    mock = MockObject(Z)
    mock.printSet('high')
    captured = capsys.readouterr()
    assert "Mock" in captured.out
    assert "zonotope" in captured.out
    assert "c: [1 2]" in captured.out 