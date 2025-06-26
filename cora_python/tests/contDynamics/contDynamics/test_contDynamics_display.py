from cora_python.contDynamics.contDynamics import ContDynamics

def test_display():
    # Create a contDynamics object
    sys = ContDynamics('test_system', 3, 1, 2, 4, 5)

    # Expected output string
    expected_str = (
        "Continuous dynamics: 'test_system'\n"
        "  number of dimensions: 3\n"
        "  number of inputs: 1\n"
        "  number of outputs: 2\n"
        "  number of disturbances: 4\n"
        "  number of noises: 5"
    )

    # Test the display method
    assert sys.display() == expected_str
    
    # Test the __str__ dunder method
    assert str(sys) == expected_str
    
    # Test the __repr__ dunder method
    assert repr(sys) == expected_str 