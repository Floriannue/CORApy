import pytest
import numpy as np
from cora_python.g.functions.verbose.print.printStruct import printStruct
from cora_python.contSet.interval.interval import Interval

def test_printStruct(capsys):
    # Test with an empty dict
    printStruct({})
    captured = capsys.readouterr()
    assert "empty struct" in captured.out

    # Test with a simple dict
    S = {'a': 1, 'b': 'text', 'c': np.array([[1, 2], [3, 4]]), 'S': Interval(2, 3)}
    printStruct(S)
    captured = capsys.readouterr()
    assert "a: 1" in captured.out
    assert "b: 'text'" in captured.out
    assert "c: [1 2; 3 4]" in captured.out
    assert "S: interval" in captured.out

    # Test with a list of dicts (struct array)
    S_list = [S, S]
    printStruct(S_list)
    captured = capsys.readouterr()
    assert "1x2 struct array with fields" in captured.out
    assert "a" in captured.out
    assert "b" in captured.out
    assert "c" in captured.out
    assert "S" in captured.out

    # Test with different accuracies
    S_float = {'val': 1.12345}
    printStruct(S_float, 'high')
    captured = capsys.readouterr()
    assert str(1.12345) in captured.out
    
    printStruct(S_float, 'low')
    captured = capsys.readouterr()
    assert "1.1" in captured.out

    # Test compact printing
    S_large_fields = {f'field_{i}': i for i in range(30)}
    printStruct(S_large_fields, 'high', do_compact=True)
    captured = capsys.readouterr()
    # In compact mode for a struct with many fields, it should summarize
    assert "struct with 30 fields" in captured.out

    printStruct(S_large_fields, 'high', do_compact=False)
    captured = capsys.readouterr()
    # Not compact, should print all fields
    assert "field_29" in captured.out 