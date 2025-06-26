import pytest

@pytest.fixture
def mock_type(monkeypatch):
    """Fixture to mock the built-in type() function"""
    original_type = type
    
    def mocked_type(obj):
        # Check if the object is an instance of a class from the test file
        if 'MockContSet' in str(original_type(obj)):
            class MockType:
                __name__ = obj._class_name
            return MockType
        return original_type(obj)

    monkeypatch.setattr('builtins.type', mocked_type) 