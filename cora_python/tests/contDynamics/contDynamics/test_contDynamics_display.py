from cora_python.contDynamics.contDynamics import ContDynamics

def test_display(ConcreteContDynamics):
    # Create a contDynamics object
    sys = ConcreteContDynamics('test_system', 3, 1, 2, 4, 5)

    # Get display string
    disp_str = sys.display()

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
    assert disp_str == expected_str
    
    # Test the __str__ dunder method
    assert str(sys) == expected_str
    
    # Test the __repr__ dunder method
    expected_repr = "ConcreteContDynamics(name='test_system', states=3, inputs=1, outputs=2, dists=4, noises=5)"
    assert repr(sys) == expected_repr 